#!/usr/bin/env python3
"""
publisher_ws.py — Jetson WebRTC publisher for DepthAI cameras.

Connects to the signaling server, registers as a publisher, and creates
a per-viewer RTCPeerConnection with the DepthAI video track attached.

Usage:
    python3 publisher_ws.py \
        --signal-url wss://example.com/ws/webrtc \
        --camera-id cam-nicu-01 \
        --token <publisher-jwt>

Dependencies:
    pip install websockets aiortc depthai numpy av
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from fractions import Fraction
from typing import Dict, Optional

import numpy as np

try:
    import depthai as dai
except ImportError:
    dai = None
    print("WARNING: depthai not installed — using test pattern source", file=sys.stderr)

import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCConfiguration, RTCIceServer
from aiortc.mediastreams import MediaStreamTrack
from av import VideoFrame

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("publisher")


# ── Proxy Track ───────────────────────────────────────────────────────────────

# ── DepthAI Video Track ───────────────────────────────────────────────────────

class DepthAIVideoTrack(MediaStreamTrack):
    """
    A video track that reads frames from a DepthAI camera pipeline.
    Falls back to a generated test pattern if DepthAI is not available.
    
    IMPORTANT: stop() is intentionally a no-op so that pc.close() does
    not kill the camera. Use force_stop() for real shutdown.
    """

    kind = "video"

    def __init__(self, fps: int = 30, width: int = 1280, height: int = 720):
        super().__init__()
        self._fps = fps
        self._width = width
        self._height = height
        self._start = time.time()
        self._frame_count = 0
        self._time_base = Fraction(1, fps)
        self._queue = None
        self._pipeline = None
        self._device = None

        if dai is not None:
            self._init_depthai()
        else:
            log.warning("DepthAI unavailable — using test pattern")

    def _init_depthai(self):
        """Initialize the DepthAI pipeline with a color camera."""
        pipeline = dai.Pipeline()

        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(self._width, self._height)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(self._fps)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("video")
        cam_rgb.preview.link(xout.input)

        self._pipeline = pipeline
        self._device = dai.Device(pipeline)
        self._queue = self._device.getOutputQueue(
            name="video", maxSize=4, blocking=False
        )
        log.info(
            "DepthAI pipeline started (%dx%d @ %d fps)",
            self._width,
            self._height,
            self._fps,
        )

    def _generate_test_frame(self):
        """Generate a color-shifting test pattern frame."""
        t = time.time() - self._start
        hue = int((t * 30) % 180)

        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        # Gradient bars
        bar_h = self._height // 8
        for i in range(8):
            color_val = (hue + i * 22) % 256
            y_start = i * bar_h
            y_end = y_start + bar_h
            frame[y_start:y_end, :, 0] = color_val
            frame[y_start:y_end, :, 1] = (color_val + 85) % 256
            frame[y_start:y_end, :, 2] = (color_val + 170) % 256

        return frame

    async def recv(self):
        """Produce the next video frame."""
        pts = self._frame_count
        self._frame_count += 1

        # Pace to target fps
        target_time = self._start + (pts / self._fps)
        wait = target_time - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        if self._queue is not None:
            # Read from DepthAI
            in_frame = self._queue.tryGet()
            if in_frame is not None:
                frame_data = in_frame.getCvFrame()
            else:
                # No frame ready — generate fallback
                frame_data = self._generate_test_frame()
        else:
            frame_data = self._generate_test_frame()

        video_frame = VideoFrame.from_ndarray(frame_data, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = self._time_base

        return video_frame

    def stop(self):
        """No-op: prevents pc.close() from killing the camera.
        Use force_stop() for real shutdown."""
        pass

    def force_stop(self):
        """Actually release DepthAI resources (called on shutdown only)."""
        super().stop()
        if self._device is not None:
            self._device.close()
            self._device = None
            log.info("DepthAI device closed")


# ── Publisher Client ──────────────────────────────────────────────────────────

class PublisherClient:
    """
    Manages the WebSocket connection to the signaling server and
    per-viewer RTCPeerConnections.
    """

    def __init__(self, signal_url: str, camera_id: str, token: str,
                 ice_servers: Optional[list] = None, force_relay: bool = False):
        self.signal_url = signal_url
        self.camera_id = camera_id
        self.token = token
        self.ws = None
        self.peers: Dict[str, RTCPeerConnection] = {}  # viewerId → pc
        self.video_track = DepthAIVideoTrack()
        self._running = True

        # Build RTCConfiguration
        rtc_ice_servers = []
        for s in (ice_servers or []):
            rtc_ice_servers.append(RTCIceServer(
                urls=s["urls"],
                username=s.get("username", ""),
                credential=s.get("credential", ""),
            ))
        self._rtc_config = RTCConfiguration(
            iceServers=rtc_ice_servers or None,
        )
        self._force_relay = force_relay
        if force_relay:
            log.info("Force-relay mode enabled — all traffic will use TURN")

    async def connect(self):
        """Connect to the signaling server and run the message loop."""
        # Ensure the path ends with /ws/webrtc
        url = self.signal_url.rstrip("/")
        if not url.endswith("/ws/webrtc"):
            url += "/ws/webrtc"

        log.info("Connecting to %s", url)

        try:
            async with websockets.connect(url) as ws:
                self.ws = ws
                log.info("WebSocket connected")

                # Register as publisher
                await self._send({
                    "type": "register-publisher",
                    "cameraId": self.camera_id,
                    "publisherToken": self.token,
                })

                async for raw in ws:
                    if not self._running:
                        break
                    await self._handle_message(raw)

        except websockets.exceptions.ConnectionClosed as e:
            log.warning("WebSocket closed: code=%s reason=%s", e.code, e.reason)
        except Exception as e:
            log.error("Connection error: %s", e)
        finally:
            await self._cleanup()

    async def _send(self, payload: dict):
        """Send a JSON message over the WebSocket."""
        if self.ws:
            msg = json.dumps(payload)
            await self.ws.send(msg)
            log.debug("Sent: %s", payload.get("type"))

    async def _handle_message(self, raw: str):
        """Dispatch incoming signaling messages.
        
        IMPORTANT: All exceptions are caught here so that one viewer's
        error never crashes the publisher's WebSocket loop.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.error("Received invalid JSON: %s", raw[:100])
            return

        msg_type = data.get("type")
        log.info("Received: %s", msg_type)

        try:
            if msg_type == "registered":
                log.info(
                    "Registered as %s for camera %s",
                    data.get("role"),
                    data.get("cameraId"),
                )

            elif msg_type == "viewer-joined":
                viewer_id = data["viewerId"]
                camera_id = data["cameraId"]
                log.info("Viewer joined: %s (camera: %s)", viewer_id, camera_id)
                await self._create_offer(viewer_id)

            elif msg_type == "answer":
                viewer_id = data["viewerId"]
                sdp = data["sdp"]
                await self._handle_answer(viewer_id, sdp)

            elif msg_type == "ice":
                viewer_id = data["viewerId"]
                candidate_data = data.get("candidate")
                await self._handle_ice(viewer_id, candidate_data)

            elif msg_type == "viewer-left":
                viewer_id = data["viewerId"]
                log.info("Viewer left: %s", viewer_id)
                await self._close_peer(viewer_id)

            elif msg_type == "error":
                log.error("Server error: %s", data.get("message"))

            else:
                log.warning("Unknown message type: %s", msg_type)

        except Exception as e:
            log.error("Error handling '%s' message: %s", msg_type, e, exc_info=True)

    async def _create_offer(self, viewer_id: str):
        """Create a new PeerConnection for a viewer and send an offer."""
        # Close any existing PC for this viewer (reconnect case)
        await self._close_peer(viewer_id)

        try:
            pc = RTCPeerConnection(configuration=self._rtc_config)
            self.peers[viewer_id] = pc

            # Add the video track directly — stop() is a no-op on
            # DepthAIVideoTrack, so pc.close() won't kill the camera.
            pc.addTrack(self.video_track)

            # ICE candidate handler — trickle to the viewer
            @pc.on("icecandidate")
            async def on_ice(candidate):
                if candidate:
                    try:
                        await self._send({
                            "type": "ice",
                            "cameraId": self.camera_id,
                            "viewerId": viewer_id,
                            "candidate": {
                                "candidate": candidate.candidate,
                                "sdpMid": candidate.sdpMid,
                                "sdpMLineIndex": candidate.sdpMLineIndex,
                            },
                        })
                    except Exception as e:
                        log.warning("Failed to send ICE candidate: %s", e)

            # Connection state logging
            @pc.on("connectionstatechange")
            async def on_conn_state():
                log.info(
                    "PeerConnection [%s] connectionState: %s",
                    viewer_id[:8],
                    pc.connectionState,
                )
                if pc.connectionState in ("failed", "closed"):
                    await self._close_peer(viewer_id)

            @pc.on("iceconnectionstatechange")
            async def on_ice_state():
                log.info(
                    "PeerConnection [%s] iceConnectionState: %s",
                    viewer_id[:8],
                    pc.iceConnectionState,
                )

            @pc.on("icegatheringstatechange")
            async def on_ice_gather():
                log.debug(
                    "PeerConnection [%s] iceGatheringState: %s",
                    viewer_id[:8],
                    pc.iceGatheringState,
                )

            # Create and send the offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            await self._send({
                "type": "offer",
                "cameraId": self.camera_id,
                "viewerId": viewer_id,
                "sdp": pc.localDescription.sdp,
            })

            log.info("Offer sent to viewer %s", viewer_id[:8])

        except Exception as e:
            log.error("Failed to create offer for viewer %s: %s", viewer_id[:8], e)
            await self._close_peer(viewer_id)

    async def _handle_answer(self, viewer_id: str, sdp: str):
        """Set the remote description from a viewer's answer."""
        pc = self.peers.get(viewer_id)
        if not pc:
            log.warning("Answer for unknown viewer %s", viewer_id[:8])
            return

        answer = RTCSessionDescription(sdp=sdp, type="answer")
        await pc.setRemoteDescription(answer)
        log.info("Remote description set for viewer %s", viewer_id[:8])
    
    async def _handle_ice(self, viewer_id, candidate_data):
        """Add an ICE candidate from a viewer (best-effort)."""
        pc = self.peers.get(viewer_id)
        if not pc or not candidate_data:
            return

        candidate_str = candidate_data.get("candidate", "")
        if not candidate_str:
            return

        # Best-effort: try to parse and add, but don't block if it fails.
        # The connection can still establish via SDP-embedded candidates.
        try:
            parts = candidate_str.split()
            if len(parts) < 8:
                return
            foundation = parts[0].split(":", 1)[-1] if ":" in parts[0] else parts[0]
            candidate = RTCIceCandidate(
                component=int(parts[1]),
                foundation=foundation,
                ip=parts[4],
                port=int(parts[5]),
                priority=int(parts[3]),
                protocol=parts[2],
                type=parts[7],
                sdpMid=candidate_data.get("sdpMid", ""),
                sdpMLineIndex=candidate_data.get("sdpMLineIndex", 0),
            )
            await pc.addIceCandidate(candidate)
            log.debug("Added ICE candidate from viewer %s", viewer_id[:8])
        except Exception:
            pass  # Silently ignore — connection works via SDP candidates

    # async def _handle_ice(self, viewer_id: str, candidate_data: dict):
    #     """Add an ICE candidate from a viewer."""
    #     pc = self.peers.get(viewer_id)
    #     if not pc:
    #         log.warning("ICE for unknown viewer %s", viewer_id[:8])
    #         return

    #     if not candidate_data:
    #         return

    #     candidate_str = candidate_data.get("candidate", "")
    #     if not candidate_str:
    #         return

    #     try:
    #         # Parse the browser's candidate SDP string into aiortc fields.
    #         # Format: "candidate:<foundation> <component> <protocol> <priority> <ip> <port> typ <type> [raddr <addr> rport <port>]"
    #         parts = candidate_str.split()
    #         if len(parts) < 8:
    #             log.warning("Malformed ICE candidate: %s", candidate_str[:80])
    #             return

    #         # parts[0] = "candidate:<foundation>" or just "<foundation>"
    #         foundation = parts[0].split(":", 1)[-1] if ":" in parts[0] else parts[0]
    #         component = int(parts[1])
    #         protocol = parts[2]
    #         priority = int(parts[3])
    #         ip = parts[4]
    #         port = int(parts[5])
    #         # parts[6] = "typ"
    #         candidate_type = parts[7]

    #         related_address = None
    #         related_port = None
    #         tcp_type = None

    #         i = 8
    #         while i < len(parts):
    #             if parts[i] == "raddr" and i + 1 < len(parts):
    #                 related_address = parts[i + 1]
    #                 i += 2
    #             elif parts[i] == "rport" and i + 1 < len(parts):
    #                 related_port = int(parts[i + 1])
    #                 i += 2
    #             elif parts[i] == "tcptype" and i + 1 < len(parts):
    #                 tcp_type = parts[i + 1]
    #                 i += 2
    #             else:
    #                 i += 1

    #         candidate = RTCIceCandidate(
    #             component=component,
    #             foundation=foundation,
    #             ip=ip,
    #             port=port,
    #             priority=priority,
    #             protocol=protocol,
    #             type=candidate_type,
    #             related_address=related_address,
    #             related_port=related_port,
    #             tcpType=tcp_type,
    #             sdpMid=candidate_data.get("sdpMid", ""),
    #             sdpMLineIndex=candidate_data.get("sdpMLineIndex", 0),
    #         )
    #         await pc.addIceCandidate(candidate)
    #         log.debug("Added ICE candidate from viewer %s: %s %s:%d",
    #                    viewer_id[:8], candidate_type, ip, port)
    #     except Exception as e:
    #         log.warning("Failed to add ICE candidate: %s", e)

    async def _close_peer(self, viewer_id: str):
        """Close and remove a peer connection."""
        pc = self.peers.pop(viewer_id, None)
        if pc:
            await pc.close()
            log.info("PeerConnection closed for viewer %s", viewer_id[:8])

    async def _cleanup(self):
        """Close all peer connections (but keep the video track alive for reuse)."""
        log.info("Cleaning up %d peer connection(s)", len(self.peers))
        for viewer_id in list(self.peers.keys()):
            await self._close_peer(viewer_id)
        self.ws = None

    def stop(self):
        """Signal the client to stop."""
        self._running = False

    def shutdown(self):
        """Full shutdown: stop the video track."""
        self._running = False
        self.video_track.force_stop()


# ── Auto-reconnect wrapper ────────────────────────────────────────────────────

async def run_with_reconnect(signal_url: str, camera_id: str, token: str,
                             ice_servers: Optional[list] = None,
                             force_relay: bool = False):
    """Run the publisher with automatic reconnection on disconnect."""
    backoff = 1
    max_backoff = 30

    client = PublisherClient(
        signal_url, camera_id, token,
        ice_servers=ice_servers,
        force_relay=force_relay,
    )

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, client.shutdown)

    while client._running:
        await client.connect()

        if not client._running:
            log.info("Shutdown requested — exiting")
            break

        log.info("Reconnecting in %ds…", backoff)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Jetson WebRTC publisher — streams DepthAI camera via signaling server",
    )
    parser.add_argument(
        "--signal-url",
        required=True,
        help="WebSocket URL for the signaling server, e.g. wss://example.com/ws/webrtc",
    )
    parser.add_argument(
        "--camera-id",
        required=True,
        help="Unique camera identifier, e.g. cam-nicu-01",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Publisher JWT token for authentication",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--turn-url",
        default=None,
        help="TURN server URL, e.g. turn:turn.example.com:3478",
    )
    parser.add_argument(
        "--turn-username",
        default="",
        help="TURN server username",
    )
    parser.add_argument(
        "--turn-credential",
        default="",
        help="TURN server credential/password",
    )
    parser.add_argument(
        "--force-relay",
        action="store_true",
        help="Force all traffic through TURN relay (debug mode)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build ICE servers list
    ice_servers = [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
    ]
    if args.turn_url:
        ice_servers.append({
            "urls": args.turn_url,
            "username": args.turn_username,
            "credential": args.turn_credential,
        })
        log.info("TURN server: %s", args.turn_url)

    log.info("Starting publisher for camera '%s'", args.camera_id)
    log.info("Signaling URL: %s", args.signal_url)
    log.info("ICE servers: %d configured", len(ice_servers))

    asyncio.run(run_with_reconnect(
        args.signal_url, args.camera_id, args.token,
        ice_servers=ice_servers,
        force_relay=args.force_relay,
    ))


if __name__ == "__main__":
    main()