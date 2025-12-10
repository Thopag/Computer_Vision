import os
import cv2
import numpy as np
from ...CONFIG import *
from ..box import xyxy_to_cxcywh, draw_box
from ultralytics import YOLO

model = YOLO(MODEL_PATH)

class yolo_detector:

    def __init__(self, blacklist = [], confidence_threshold = 0.0):

        self.last_result = None
        self.blacklist = blacklist
        self.confidence_threshold = confidence_threshold

        self.boxes_xy = []
        self.boxes_cc = []
        self.labels = []
        self.confs = []

    def detect(self, frame_bgr):

        self.boxes_xy.clear()
        self.boxes_cc.clear()
        self.labels.clear()
        self.confs.clear()

        h, w = frame_bgr.shape[:2]

        # Convert image to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results =  model(frame_rgb, verbose= False)[0]
        class_names = results.names

        boxes = results.boxes
        for box in boxes:

            # Check confidence
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue

            # Check label
            class_id = int(box.cls[0])
            label = class_names[class_id]
            if label in self.blacklist:
                continue

            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_xy = (x1, y1, x2, y2)
            box_cc = xyxy_to_cxcywh(box_xy)
    
            self.confs.append(conf)
            self.labels.append(label)
            self.boxes_xy.append(box_xy)
            self.boxes_cc.append(box_cc)

        self.last_results = results

        return self.boxes_cc, self.labels

    def have_something(self):

        if len(self.labels) == 0:
            return False

        return True

    def draw_detections(self, frame):

        if self.have_something():
            color = [255,0,0]
            for box, label, conf in zip(self.boxes_cc, self.labels, self.confs):
                label = f"{label} {conf:.2f}"
                draw_box(frame, box, color, label)

        return
