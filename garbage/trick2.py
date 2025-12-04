# This file contains the definition of the trick2 function 
import cv2
import numpy as np 
from utils.yolo import detect_objects
from utils.color import detect_color, change_color_mask
from utils.ROI import roi
from utils.geometry import grow_object

def trick2(cap: cv2.VideoCapture, writer: cv2.VideoWriter ,
            nb_frame , to_remove ,read_every_x_frame, file=None):
    switch = 0
    overlap_prev = False

    last_masks = []
    last_labels = []

    bottle_mask = None

    file.write(f"Start trick 2 \n")
    
    # Labels you want to detect
    target_labels = ["bottle", "cell_phone"]

    for frame_idx in range(nb_frame):
        ok, frame = cap.read()
        if not ok:
            break

        print(f"Trick 2 progress: {(frame_idx)/(nb_frame) *100:.2f} %", end="\r")
        file.write(f"Trick 2 progress: {(frame_idx)/(nb_frame) *100:.2f} %\n")

        output = frame
        #-------------------------------
        # Color Detection (Red) WAND
        #-------------------------------
        mask_red  = detect_color(frame, [255,0,5], [55,55],[255,255], tuning=25)
        roi_red  = roi(mask_red, 3, 20)
        #-------------------------------
        # color detection Blue  ball 
        #------------------------------- 
        mask_blue = detect_color(frame, [0,0,255], [100,100],[255,255], tuning=25)
        roi_blue = roi(mask_blue, 10, 20)
        #-------------------------------
        # Object Detection and Filtering
        #-------------------------------
        if frame_idx % read_every_x_frame == 0:
            last_masks, last_labels = detect_objects(frame)
            filtered_masks  = []
            filtered_labels = []
            for mask, label in zip(last_masks, last_labels):
                if label not in to_remove:
                    filtered_masks.append(mask)
                    filtered_labels.append(label)

            last_masks  = filtered_masks
            last_labels = filtered_labels
        # -------------------------------
        # Draw bounding boxes for each object
        # -------------------------------
        
        if not (last_masks == []):
            for mask, label in zip(last_masks, last_labels):

                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()

                    cv2.rectangle(output,
                                  (x_min, y_min),
                                  (x_max, y_max),
                                  (0, 255, 0),
                                  2)
                    cv2.putText(output, str(label),
                                (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

        # -------------------------------
        # wand logic here
        # -------------------------------
        # first add blue ball masks
        last_masks.append(roi_blue)
        last_labels.append("blue ball")
        global_overlap = np.zeros_like(roi_red, dtype=np.uint8)

        for mask in last_masks:
                overlap = cv2.bitwise_and(mask, roi_red)
                global_overlap = cv2.bitwise_or(global_overlap, overlap)
        
        overlap_pixels = np.count_nonzero(global_overlap)

        # ----- STABLE EVENT DETECTION -----
        overlap_now = overlap_pixels > 30   # threshold

        if overlap_now and not overlap_prev:
            # A NEW overlap event occurred here
            switch += 1

        overlap_prev = overlap_now

        # ----- STATE LOGIC -----
        state = switch >= 1    # state becomes True after first switch
        
        #------------------------
        # trick logic 
        #-----------------------
        
        mask_blue = roi(mask_blue, 10, 15)
        if switch == 1:
            output = change_color_mask(frame, [0,0,255],[255,0,0] , mask_blue)
        elif switch ==2 :
            output = change_color_mask(frame, [0,0,255],[0,255,0] , mask_blue)
        elif switch == 3 :
            output = frame 
        elif switch == 4:
            # Default: no object found
            target_index = -2
            # Find the FIRST matching label in the list
            for i, lab in enumerate(last_labels):
                if lab in target_labels:
                    target_index = i
                    break

            # If object found
            if target_index != -2:
                object_mask = last_masks[target_index]
                output = grow_object(frame, object_mask, scale=3.0)
            else:
                # No bottle or cellphone found → keep frame unchanged
                #print("No bottle/cell_phone detected in this frame.")
                output = frame
                
        else : #switch == 5:
            output = frame

        # ----- Debug -----
        cv2.putText(output, f"Overlap: {overlap_pixels}",
                    (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)
        cv2.putText(output, f"Switch: {switch}",
                    (50,90), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
        cv2.putText(output,f"State: {state}",
                    (50,130), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

        writer.write(output)
    
    file.write(f"End trick 2 \n")

    return 1