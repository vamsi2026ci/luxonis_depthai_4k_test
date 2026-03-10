import depthai as dai
import cv2

# 1) Build pipeline
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 360)
cam.setInterleaved(False)
cam.setFps(30)

xout = pipeline.createXLinkOut()
xout.setStreamName("rgb")
cam.preview.link(xout.input)

# 2) Connect to device + start
with dai.Device(pipeline) as device:
    q = device.getOutputQueue("rgb", maxSize=4, blocking=False)

    while True:
        msg = q.get()                 # dai.ImgFrame
        frame = msg.getCvFrame()      # numpy BGR
        cv2.imshow("rgb", frame)
        if cv2.waitKey(1) == ord('q'):
            break