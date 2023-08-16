# This module detects any movement in the area with ROI
# env:area_env

import sys
import os
import cv2
from tracker import *


args: list[int] = sys.argv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

sysIndex = sys.argv[1]
print("sysIndex: " + sysIndex)
cam_capture = cv2.VideoCapture(sysIndex)

frame_width = int(cam_capture.get(3))
frame_height = int(cam_capture.get(4))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
videoWriter = cv2.VideoWriter('output.avi', fourcc, 24, (frame_width, frame_height))
cv2.destroyAllWindows()

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
tracker = EuclideanDistTracker()

ids = []
counter = 0

while True:
    flag, im0 = cam_capture.read()
    if flag:
        showCrosshair = False
        fromCenter = False
        r = cv2.selectROI("Image", im0, fromCenter, showCrosshair)
        break

while True:
    _, image_frame = cam_capture.read(0)
    rect_img = image_frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    mask = object_detector.apply(rect_img)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)

        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    check = False

    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(rect_img, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(rect_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        for i in ids:
            if i == box_id:
                check = True

    counter = counter + 1
    cv2.rectangle(image_frame, (int(r[0]), int(r[1])), (int(r[0] + r[2]), int(r[1] + r[3])), (255, 255, 0), 3)

    # Change the image drawn in ROI
    image_frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])] = rect_img
    cv2.imshow("Sketcher ROI", image_frame)
    videoWriter.write(image_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_capture.release()
cv2.destroyAllWindows()
