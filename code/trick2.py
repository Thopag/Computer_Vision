# This file contains functions to do the trick 2

import cv2 as cv 
from helper import get_trick_start_frames

# Path to input video, output video, and JSON annotation file
in_path  = "Computer_Vision/data/group_11_fixed.mp4"    
out_path = "Computer_Vision/output/t2/group_11_fixed_t2_test1.mp4"
json_path = "Computer_Vision/data/annotations_group_11.json"

cap = cv.VideoCapture(in_path)
if not cap.isOpened():  
    raise IOError(f"Could not open {in_path}")

fps = cap.get(cv.CAP_PROP_FPS) # Should be used to pass it to get_trick_start_frames
print(f"Video FPS: {fps}")
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

out = cv.VideoWriter(out_path,
                      cv.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))
