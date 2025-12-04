#!/usr/bin/env python3
import cv2
import os
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import time
from trick1 import trick1
from garbage.trick2 import trick2
from utils.video_splitting import get_number_of_frames
from CONFIG import *

#----------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#------------------------------------------------
grp = 11
fixed = False

if fixed:
    t = "fixed"
else:
    t = "dynamic"

in_path  = f"../input/video_group_{grp}_{t}.mp4"
out_path = f"../output/main_group_{grp}_{t}.mp4"
txt_path = out_path[:-4] + ".txt"

json_path = f"../input/annotations_group_{grp}.json"

def main() :
    start_main = time.time()
    cap = cv2.VideoCapture(in_path)

    f = open(txt_path, "w", encoding="utf-8")

    #------------General set up------------#

    if not cap.isOpened():
        raise IOError(f"Could not open {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    start = time.time()
    nbr_trick1, nbr_trick2, nbr_trick3 = get_number_of_frames(cap, json_path, fixed)

    #------------Trick1------------#
    start_1 = time.time()

    blacklist = ['person', 'skateboard', 'laptop', 'cup', 'chair', 'dining table', 'microwave', 'umbrella', 
    'kite', 'cat', 'traffic light', 'book', 'cell phone', 'keyboard', 'scissors', 'frisbee', 'suitcase', 'dog', 'tv']
    with_first_frame = False
    read_every_x_frame = 5
    forget_time = 2*fps

    SE_fraction=0.75
    tuning=35 # Default 35

    trick1(writer, cap, nbr_trick1, blacklist, read_every_x_frame, forget_time, f, SE_fraction, tuning, with_first_frame, grp)

    end_1 = time.time()
    #------------Trick2------------#
    start_2 = time.time()

    to_remove = ["person", "dining table", "sports ball",
                    "orange", "handbag","keyboard" ]
    read_every_x_frame = 10
    
    #trick2(cap, writer, nbr_trick2, to_remove, read_every_x_frame, file=f)

    end_2 = time.time()
    #------------Trick3------------#
    start_3 = time.time()

    #trick3(cap, writer, nbr_trick3, ....)

    end_3 = time.time()
    #------------Finish------------#
    end_main = time.time()

    cap.release()
    writer.release()

    print("=============================================")
    print("Saved:", out_path)
    elapsed_1 = end_1 - start_1
    print(f"Trick1 Execution time: {elapsed_1:.4f} seconds")
    elapsed_2 = end_2 - start_2
    print(f"Trick2 Execution time: {elapsed_2:.4f} seconds")
    elapsed_3 = end_3 - start_3
    print(f"Trick3 Execution time: {elapsed_3:.4f} seconds")
    elapsed = end_main - start_main
    print(f"The main execution time is :{elapsed:.4f}")
    print("=============================================")

    f.close()

if __name__ == "__main__":
    main()