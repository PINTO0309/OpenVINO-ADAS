#!/usr/bin/env python

import sys
import cv2
import numpy as np
from PIL import Image
import time

model_xml='lrmodels/FP16/semantic-segmentation-adas-0001.xml'
model_bin='lrmodels/FP16/semantic-segmentation-adas-0001.bin'
seg_image = Image.open("data/input/009649.png")
palette = seg_image.getpalette() # Get a color palette
camera_width = 320
camera_height = 240
fps = ""
framepos = 0
frame_count = 0
vidfps = 0
elapsedTime = 0

net = cv2.dnn.readNet(model_xml, model_bin)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

cap = cv2.VideoCapture("data/input/testvideo.mp4")
camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
vidfps = int(cap.get(cv2.CAP_PROP_FPS))
print("videosFrameCount =", str(frame_count))
print("videosFPS =", str(vidfps))

time.sleep(1)

while cap.isOpened():
    t1 = time.time()
    cap.set(cv2.CAP_PROP_POS_FRAMES, framepos)
    ret, frame = cap.read()
    if not ret:
        break
    #frame = cv2.imread('data/input/000003.jpg')
    prepimg = frame[:, :, ::-1].copy()
    prepimg = Image.fromarray(prepimg)
    prepimg = prepimg.resize((2048, 1024), Image.ANTIALIAS)
    prepimg = np.asarray(prepimg)

    t2 = time.perf_counter()
    blob = cv2.dnn.blobFromImage(prepimg)
    net.setInput(blob)
    out = net.forward()

    print(out.shape)
    sys.exit(0)

    outputs = exec_net.requests[0].outputs[out_blob] # (1, 1, 1024, 2048)
    print("SegmentationTime = {:.7f}".format(time.perf_counter() - t2))
    outputs = outputs[0][0]
    #print(outputs.shape)
    outputs = cv2.resize(outputs, (camera_width, camera_height))

    # View
    image = Image.fromarray(np.uint8(outputs), mode="P")
    image.putpalette(palette)
    image = image.convert("RGB")

    image = np.asarray(image)
    image = cv2.addWeighted(frame, 1, image, 0.9, 0)

    cv2.putText(image, fps, (camera_width-180,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
    cv2.imshow("Result", image)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break
    elapsedTime = time.time() - t1
    fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)
    framepos += vidfps

cv2.destroyAllWindows()

