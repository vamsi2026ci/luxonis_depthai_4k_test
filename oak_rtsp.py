#!/usr/bin/env python3
import subprocess
import time
import signal
import sys

import depthai as dai


RTSP_URL = "rtsp://127.0.0.1:8554/optical_2"   # your RTSP server publish URL
FPS = 5
WIDTH, HEIGHT = 1280, 720  # choose what you want


def make_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()

    cam = pipeline.createColorCamera()
    cam.setFps(FPS)
    cam.setVideoSize(WIDTH, HEIGHT)
    cam.setInterleaved(False)

    enc = pipeline.createVideoEncoder()
    enc.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.H264_BASELINE)
    cam.video.link(enc.input)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("h264")
    enc.bitstream.link(xout.input)

    return pipeline


def start_ffmpeg_rtsp() -> subprocess.Popen:
    # We send H.264 Annex-B bytestream to ffmpeg stdin, then "copy" to RTSP.
    cmd = [
        "ffmpeg",
        "-loglevel", "warning",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-f", "h264",
        "-i", "pipe:0",

        # If your RTSP server/viewer expects periodic SPS/PPS and keyframes,
        # ensure your encoder produces them. DepthAI usually does, but if not,
        # see the "force IDR / repeat headers" notes below.
        "-c:v", "copy",
        "-an",

        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        RTSP_URL,
    ]

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    ff = start_ffmpeg_rtsp()

    def shutdown(*_):
        try:
            if ff.stdin:
                ff.stdin.close()
        except Exception:
            pass
        ff.terminate()
        return

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    pipeline = make_pipeline()

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue("h264", maxSize=30, blocking=True)

        last = time.time()
        while True:
            pkt = q.get()  # encoded chunk
            data = pkt.getData().tobytes()

            if not ff.stdin:
                raise RuntimeError("ffmpeg stdin is not available")

            try:
                ff.stdin.write(data)
            except BrokenPipeError:
                raise RuntimeError("ffmpeg exited (broken pipe). Is your RTSP server running?")

            # optional: flush occasionally (not every packet)
            now = time.time()
            if now - last > 0.5:
                ff.stdin.flush()
                last = now

    return 0


if __name__ == "__main__":
    sys.exit(main())