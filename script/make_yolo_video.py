import cv2
import os

from .utils.detection.yolo_detector import yolo_detector
from .CONFIG import *


def yolo_video(in_path=IN_PATH, read_every_x_frame=READ_EVERY_X_FRAME, blacklist=BLACKLIST , confidence_threshold=CONF_TRESHOLD):

    cap = cv2.VideoCapture(in_path)
    out_path = f"files/yolo_output/{FILE_NAME}.mp4"

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

if __name__ == "__main__":

    yolo_video(IN_PATH, read_every_x_frame=READ_EVERY_X_FRAME, blacklist=[], confidence_threshold=CONF_TRESHOLD)