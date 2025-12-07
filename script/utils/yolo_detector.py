import os
import cv2
import numpy as np
from ..CONFIG import *
from .box import xyxy_to_cxcywh, draw_box
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
            
            self.confs.append(conf)
            self.labels.append(label)

            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_xy = (x1, y1, x2, y2)
            self.boxes_xy.append(box_xy)
            self.boxes_cc.append(xyxy_to_cxcywh(box_xy))

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

def yolo_video(in_path=IN_PATH, read_every_x_frame=READ_EVERY_X_FRAME, blacklist=BLACKLIST , confidence_threshold=CONF_TRESHOLD):

    cap = cv2.VideoCapture(in_path)
    file_name = os.path.basename(in_path)[:-4]
    out_path = f"files/yolo_output/{file_name}.mp4"

    txt_path = out_path[:-4] + ".txt"

    f = open(txt_path, "w", encoding="utf-8")

    f.write(f"Start yolo predictions \n")
    f.write(f"input path : {in_path} \n")
    f.write(f"output path : {out_path} \n")
    f.write(f"prediction every {read_every_x_frame} frames \n")
    f.write(f"confidence_threshold = {confidence_threshold} \n\n")

    detector = yolo_detector(blacklist, confidence_threshold)

    #------------Set up------------#

    if not cap.isOpened():
        raise IOError(f"Could not open {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    seen_labels = []

    for frame_idx in range(total_frames):

        ok, frame = cap.read()
        if not ok:
            break
        
        print(f"Progress: {(frame_idx)/(total_frames) *100:.2f} %", end="\r")

        # Make a copy to draw on
        annotated_frame = frame.copy()

        if frame_idx % read_every_x_frame == 0:

            f.write(f"\n  Video at {frame_idx/fps:.3f} s ({(frame_idx)/(total_frames) *100:.2f} %)\n")

            detector.detect(frame)

            if detector.have_something():
                for label, conf in zip(detector.labels, detector.confs):
                    if label not in seen_labels:
                            seen_labels.append(label)
                    f.write(f"Find [{label}] with confidence [{conf:.2f}]\n")
        
        # draw the boxes
        detector.draw_detections(annotated_frame)

        writer.write(annotated_frame)
    
    f.write(f" All seen labels {seen_labels}\n")
    cap.release()
    writer.release()
    f.close()
    
    return out_path