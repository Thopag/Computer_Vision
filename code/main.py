#!/usr/bin/env python3
import cv2
import os
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import time
from trick2 import trick2
from utils.video import load_json, time_to_seconds


#----------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#LOGGER.setLevel("ERROR")  # silence YOLO
#------------------------------------------------
in_path  = "data/splits_fixed/trick_2.mp4"
out_path = "output/trick_2_TEST_MAIN.mp4"

json_path = "data/annotations_group_11.json"

def main() :
    start_main = time.time()
    cap = cv2.VideoCapture(in_path)
    # Load YOLO model 
    
    if not cap.isOpened():
        raise IOError(f"Could not open {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    #------------Trick2--------------------------------------------------
    start = time.time() 
    to_remove = ["person", "dining table", "sports ball",
                    "orange", "handbag","keyboard" ]
    read_every_x_frame = 10
    
    # in second
    start_trick1 = 0
    tmp= load_json(json_path , True , 2)
    start_trick2 =time_to_seconds(tmp)
    end_trick1 = start_trick2
    tmp =load_json(json_path , True , 3)
    end_trick2 = time_to_seconds(tmp)
    start_trick3 = end_trick2
    
    # in frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_trick1 = int(start_trick1*fps)
    start_trick2 = int(start_trick2*fps)
    start_trick3 = int(start_trick3*fps)
    end_trick1   = int(end_trick1*fps)
    end_trick2   = int(end_trick2*fps)
    end_trick3   = int(total_frames)
    
    trick2(cap, writer ,end_trick2-start_trick2, to_remove ,read_every_x_frame)
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