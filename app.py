import numpy as np
import cv2
import cfg
import time

cap = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromCaffe(cfg.prototxt, cfg.caffemodel)
start_time = time.time()
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    cv2.waitKey(1000)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.9:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        areaX = int(endX - startX)
        areaY = int(endY - startY)
        area = (areaX * areaY)
        if area > 25000:
            start_time = time.time()
            cv2.namedWindow('Face Detected', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Face Detected', 500, 500)
            cv2.imshow('Face Detected', frame)
            cv2.waitKey(1)
        else:
            if time.time() - start_time > 1:
                cv2.destroyAllWindows()
    if time.time() - start_time > 1:
        cv2.destroyAllWindows()
