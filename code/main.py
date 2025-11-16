#!/usr/bin/env python3
import cv2
import os
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import time
from trick2 import trick2


#----------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
LOGGER.setLevel("ERROR")  # silence YOLO
#------------------------------------------------
in_path  = "data/splits_fixed/trick_2.mp4"
out_path = "output/trick_2_TEST_MAIN.mp4"


def main() :
    start_main = time.time()
    cap = cv2.VideoCapture(in_path)
    # Load YOLO model 
    model = YOLO('yolov8n.pt')  # Load the model ONCE
    if not cap.isOpened():
        raise IOError(f"Could not open {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    start = time.time() 
    trick2(cap, writer ,model )
    end = time.time()
    cap.release()
    writer.release()
    end_main = time.time()
    print("=============================================")
    print("Saved:", out_path)
    elapsed = end -start
    print(f"Trick2 Execution time: {elapsed:.4f} seconds")
    elapsed = end_main - start_main
    print(f"The main execution time is :{elapsed:.4f}")
    print("=============================================")

if __name__ == "__main__":
    main()