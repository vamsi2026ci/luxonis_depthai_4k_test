#!/usr/bin/env python3
import asyncio
import json
import sys
import time
from fractions import Fraction

import depthai as dai
import av

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc import MediaStreamTrack
from av import VideoFrame


# ===================== SETTINGS =====================
FPS = 15
WIDTH, HEIGHT = 1280, 720

# IMPORTANT:
# While you validate camera + media, TURN just creates noisy failures (403 Forbidden IP).
# Leave ONLY STUN for now. Add TURN back only when you move across networks and you own the TURN server.
RTC_CONFIGURATION = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        # If you re-enable TURN later, use a TURN server you control.
        RTCIceServer(
            urls=["turn:free.expressturn.com:3478?transport=udp","turn:free.expressturn.com:3478?transport=tcp"],
            username="000000002088024667",
            credential="MjgfoDH1sScYnDE14b2AqUtH+QQ=",
        ),
    ]
)
# ====================================================


def make_depthai_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()

    cam = pipeline.createColorCamera()
    cam.setFps(FPS)
    cam.setVideoSize(WIDTH, HEIGHT)
    cam.setInterleaved(False)

    enc = pipeline.createVideoEncoder()
    enc.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.H264_BASELINE)

    # Make decoder startup reliable: force frequent IDR
    try:
        enc.setKeyframeFrequency(FPS)  # ~1 keyframe/sec
    except Exception:
        pass

    cam.video.link(enc.input)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("h264")
    enc.bitstream.link(xout.input)

    return pipeline


class DepthAIH264DecodedTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, device: dai.Device, queue_name: str = "h264"):
        super().__init__()
        self._q = device.getOutputQueue(queue_name, maxSize=30, blocking=True)
        self._decoder = av.CodecContext.create("h264", "r")

        self._frame_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=60)
        self._running = True

        self._frame_index = 0
        self._time_base = Fraction(1, FPS)

        self._decode_count = 0
        self._sent_count = 0

        self._worker_task = asyncio.create_task(self._decode_worker())

    async def _decode_worker(self):
        loop = asyncio.get_event_loop()
        while self._running:
            pkt = await loop.run_in_executor(None, self._q.get)
            if not self._running:
                break

            data = pkt.getData().tobytes()

            try:
                avpkt = av.packet.Packet(data)
                frames = self._decoder.decode(avpkt)
            except Exception:
                continue

            for f in frames:
                if not self._running:
                    break

                arr = f.to_ndarray(format="bgr24")
                vf = VideoFrame.from_ndarray(arr, format="bgr24")

                vf.pts = self._frame_index
                vf.time_base = self._time_base
                self._frame_index += 1

                self._decode_count += 1
                if self._decode_count % 30 == 0:
                    print(f"[DepthAI] decoded frames: {self._decode_count}", flush=True)

                if self._frame_queue.full():
                    try:
                        _ = self._frame_queue.get_nowait()
                    except Exception:
                        pass

                await self._frame_queue.put(vf)

    async def recv(self) -> VideoFrame:
        if not self._running:
            raise asyncio.CancelledError

        frame = await self._frame_queue.get()

        self._sent_count += 1
        if self._sent_count % 30 == 0:
            print(f"[aiortc] sent frames: {self._sent_count}", flush=True)

        return frame

    # MUST be sync for aiortc
    def stop(self):
        self._running = False
        try:
            if self._worker_task:
                self._worker_task.cancel()
        except Exception:
            pass
        super().stop()


async def read_json_from_stdin(prompt: str):
    print(prompt, flush=True)
    line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


def sdp_has_candidates(sdp: str) -> bool:
    return "a=candidate:" in sdp


async def main():
    pc = RTCPeerConnection(RTC_CONFIGURATION)
    track = None

    @pc.on("icecandidate")
    def on_icecandidate(candidate):
        # Optional: print Jetson trickle candidates (often already in SDP)
        if candidate is None:
            print("ICE gathering complete on Jetson", flush=True)
            return
        out = {
            "candidate": candidate.candidate,
            "sdpMid": candidate.sdpMid,
            "sdpMLineIndex": candidate.sdpMLineIndex,
        }
        print("JETSON ICE (optional):", flush=True)
        print(json.dumps(out), flush=True)

    try:
        pipeline = make_depthai_pipeline()

        with dai.Device(pipeline) as device:
            track = DepthAIH264DecodedTrack(device, "h264")
            pc.addTrack(track)

            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            offer_json = {"type": pc.localDescription.type, "sdp": pc.localDescription.sdp}
            print("\nJETSON OFFER -> paste into browser applyOfferFromJetson(...):", flush=True)
            print(json.dumps(offer_json), flush=True)

            ans = await read_json_from_stdin("\nPaste BROWSER ANSWER JSON here, then press Enter:")
            if ans is None:
                return

            await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
            print("Jetson set remote description (answer).", flush=True)

            # If the answer SDP already contains candidates, DO NOT ask for ICE.
            if sdp_has_candidates(ans["sdp"]):
                print("\nBrowser answer SDP already contains ICE candidates.", flush=True)
                print("No need to paste browser ICE. Leaving connection running. Ctrl+C to stop.\n", flush=True)
                while True:
                    await asyncio.sleep(1)

            # Otherwise (rare): accept trickled ICE candidates
            print("\nAnswer SDP had no candidates. Paste BROWSER ICE candidates one-by-one. Ctrl+C to stop.", flush=True)
            while True:
                cand = await read_json_from_stdin("Paste one BROWSER ICE JSON candidate:")
                if cand is None:
                    continue

                # Robust validation: ignore junk
                if not isinstance(cand, dict) or "candidate" not in cand:
                    print("Ignored input (not an ICE candidate JSON with 'candidate' field).", flush=True)
                    continue

                await pc.addIceCandidate(
                    RTCIceCandidate(
                        candidate=cand["candidate"],
                        sdpMid=cand.get("sdpMid", None),
                        sdpMLineIndex=cand.get("sdpMLineIndex", None),
                    )
                )
                print("Added browser ICE candidate to Jetson.", flush=True)

    finally:
        try:
            if track:
                track.stop()
        except Exception:
            pass
        try:
            await pc.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass