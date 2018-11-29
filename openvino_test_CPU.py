#!/usr/bin/env python

import sys
import cv2
import numpy as np
from PIL import Image
import time
from openvino.inference_engine import IENetwork, IEPlugin

model_xml='lrmodels/FP32/semantic-segmentation-adas-0001.xml'
model_bin='lrmodels/FP32/semantic-segmentation-adas-0001.bin'
net = IENetwork.from_ir(model=model_xml, weights=model_bin)
seg_image = Image.open("data/input/009649.png")
palette = seg_image.getpalette() # Get a color palette
camera_width = 320
camera_height = 240
fps = ""
elapsedTime = 0

#plugin = IEPlugin(device="HETERO:MYRIAD,CPU")
#plugin.set_config({"TARGET_FALLBACK": "HETERO:MYRIAD,CPU"})
#plugin.set_initial_affinity(net)

#plugin = IEPlugin(device="MYRIAD")

plugin = IEPlugin(device="CPU")

plugin.add_cpu_extension("/home/alpha/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so")
exec_net = plugin.load(network=net)

input_blob = next(iter(net.inputs))        #input_blob = 'input'
out_blob   = next(iter(net.outputs))       #out_blob   = 'output/BiasAdd'
n, c, h, w = net.inputs[input_blob].shape  #n, c, h, w = 1, 3, 256, 256

del net

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
time.sleep(1)

while cap.isOpened():
    t1 = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    #frame = cv2.imread('data/input/000003.jpg')
    prepimg = frame[:, :, ::-1].copy()
    prepimg = Image.fromarray(prepimg)
    prepimg = prepimg.resize((2048, 1024), Image.ANTIALIAS)
    prepimg = np.asarray(prepimg) / 255.0
    prepimg = prepimg.transpose((2, 0, 1)).reshape((1, c, h, w))

    t2 = time.perf_counter()
    exec_net.start_async(request_id=0, inputs={input_blob: prepimg})

    if exec_net.requests[0].wait(-1) == 0:
        outputs = exec_net.requests[0].outputs[out_blob] # (1, 3, 2048, 1024)
        print("SegmentationTime = {:.7f}".format(time.perf_counter() - t2))
        #outputs = outputs.transpose((2, 3, 1, 0)).reshape((h, w, c)) # (240, 320, 3)
        outputs = cv2.resize(outputs, (camera_width, camera_height)) # (320, 240, 3)

        # View
        image = Image.fromarray(np.uint8(outputs[0]), mode="P")
        image.putpalette(palette)
        image = image.convert("RGB")

        image = np.asarray(image)
        image = prepimg = image[:, :, ::-1].copy()
        image = cv2.addWeighted(frame, 1, image, 0.9, 0)

    cv2.putText(image, fps, (camera_width-180,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
    cv2.imshow("Result", image)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break
    elapsedTime = time.time() - t1
    fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)

cv2.destroyAllWindows()
del exec_net
del plugin
