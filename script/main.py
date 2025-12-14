#!/usr/bin/env python3
import cv2
import os
import time
from script.tricks.trick1 import trick1
from script.tricks.trick2 import trick2
from script.tricks.trick3 import trick3
from .utils.video_splitting import get_number_of_frames
from .CONFIG import *

#----------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main() :
    start_main = time.time()
    cap = cv2.VideoCapture(IN_PATH)

    #------------General set up------------#

    if not cap.isOpened():
        raise IOError(f"Could not open {IN_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, (w, h))

    start = time.time()
    nbr_trick1, nbr_trick2, nbr_trick3 = get_number_of_frames(cap, JSON_PATH, FIXED)

    interpolation_obj_txt = f"files/interpolation/{FILE_NAME}_{N_OBJECT}_obj.txt"
    interpolation_wand_2_txt = f"files/interpolation/{FILE_NAME}_wand_trick2.txt"
    interaction_txt = f"files/object_&_wand/{FILE_NAME}_interaction_object_&_wand.txt"

    interpolation_ball_txt = f"files/interpolation/{FILE_NAME}_ball.txt"
    interpolation_wand_3_txt = f"files/interpolation/{FILE_NAME}_wand_trick3.txt"

    debug = True

    #------------Trick1------------#
    print("-------- 1 --------")
    start_1 = time.time()

    trick1(writer, cap, nbr_trick1, interpolation_obj_txt, debug)

    end_1 = time.time()
    #------------Trick2------------#
    print("-------- 2 --------")
    start_2 = time.time()
    
    trick2(cap, writer, nbr_trick2, nbr_trick1,
            interpolation_obj_txt, interaction_txt, interpolation_wand_2_txt, debug)

    end_2 = time.time()
    #------------Trick3------------#
    print("-------- 3 --------")
    start_3 = time.time()

    trick3(cap, writer, nbr_trick3, nbr_trick1+nbr_trick2,
            interpolation_ball_txt, interpolation_wand_3_txt, debug=False)

    end_3 = time.time()
    #------------Finish------------#
    end_main = time.time()

    cap.release()
    writer.release()

    print("=============================================")
    print("Saved:", OUT_PATH)
    elapsed_1 = end_1 - start_1
    print(f"Trick1 Execution time: {elapsed_1:.4f} seconds")
    elapsed_2 = end_2 - start_2
    print(f"Trick2 Execution time: {elapsed_2:.4f} seconds")
    elapsed_3 = end_3 - start_3
    print(f"Trick3 Execution time: {elapsed_3:.4f} seconds")
    elapsed = end_main - start_main
    print(f"The main execution time is :{elapsed:.4f}")
    print("=============================================")


if __name__ == "__main__":
    main()