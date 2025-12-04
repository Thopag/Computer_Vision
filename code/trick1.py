import cv2
import os
import numpy as np

from utils.color import detect_color
from utils.ROI import get_overlap_componant, is_some_overlap
from utils.yolo import detect_objects

def trick1(writer, cap, nb_frame, blacklist=[], read_every_x_frame=5, forget_time = 5*60, file=None, SE_fraction=0.75, tuning=35, with_first_frame=False, grp="404"):
    """
    Writer the next nbr_frame in the cap, with thhe trick 1.
    Blacklist are the detected object that should be ignored.

    writer : writer of the output video
    cap : the cap of the video
    with_first_frame : Set to True to create the invisibility with the first frame
    file : file used as logger.txt
    """

    file.write(f"Start trick 1 \n")
    file.write(f"With first frame = {with_first_frame} \n")
    file.write(f"blacklist : {blacklist} \n")


    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_memory = cv2.VideoWriter(f"../memory/{grp}_memory.mp4", fourcc, fps, (w, h))

    ok, first_frame = cap.read()
    if not ok:
        print("no frames")

    # Use to show object kernels in memory video
    red_overlay = np.zeros_like(first_frame)
    red_overlay[:, :] = (0, 0, 255)

    # Use to show object kernels in memory video
    green_overlay = np.zeros_like(first_frame)
    green_overlay[:, :] = (0, 255, 0)

    # kernel = center on the object
    memorised_objects = []
    memorised_labels = []
    memorised_kernels = []
    forget_timer = []

    file.write(f"Adaptative fraction of s_erosion for kernels = {SE_fraction} \n")

    for frame_idx in range(nb_frame):

        ok, frame_bgr = cap.read()
        if not ok:
            break

        print(f"Trick 1 progress: {(frame_idx)/(nb_frame) *100:.2f} %. Memorised labels {memorised_labels}", end="\r")
        file.write(f"\n   Trick 1 at {frame_idx/fps:.3f} s ({(frame_idx)/(nb_frame) *100:.2f} %)\n")
        file.write(f"Memorised labels {memorised_labels} \n")
        file.write(f"Forget timers {forget_timer} \n")

        # Get the green mask
        green = [0,255,0]
        green_mask = detect_color(frame_bgr, green, [40,40], [255,255], tuning=tuning)

        cloak_mask = np.zeros_like(frame_bgr[:,:,1])
        is_overlapping = False

        if frame_idx % read_every_x_frame == 0:
            mask_objects, labels, dims = detect_objects(frame_bgr, get_dim=True)
            file.write(f"Detected labels: {labels} \n")

            # Update memory loop
            for label, mask_object, dim in zip(labels, mask_objects, dims):

                if label not in blacklist:
                    
                    kernel_er = [int(dim[0]*SE_fraction), int(dim[1]*SE_fraction)]
                    adaptive_SE = cv2.getStructuringElement(cv2.MORPH_RECT,kernel_er)

                    file.write(f"Label [{label}] ")
                    in_memory = False
                    for i in range(len(memorised_objects)):

                        # Verif if the current object is already in the memory and if yes, update it
                        # We consider it is the same object if at least 1 pixel overlap, to take account of mouvement
                        if is_some_overlap(memorised_objects[i], mask_object):
                            file.write(f"replace [{memorised_labels[i]}] (object [{i}]) \n")
                            memorised_labels[i] = label
                            memorised_objects[i] = mask_object
                            memorised_kernels[i] = cv2.erode(mask_object, adaptive_SE)
                            forget_timer[i] = forget_time
                            in_memory = True

                    if not in_memory:
                        file.write(f"is a new label\n")
                        memorised_labels.append(label)
                        memorised_objects.append(mask_object)
                        memorised_kernels.append(cv2.erode(mask_object, adaptive_SE))
                        forget_timer.append(forget_time)

        # Search overlapping loop
        for i, memorised_kernel in enumerate(memorised_kernels):

            # Check if overlap
            overlapping, overlap_mask = get_overlap_componant(green_mask, memorised_kernel)

            # Add the componant into the cloak (invisibility part)
            if overlapping:
                cloak_mask = cv2.add(cloak_mask, overlap_mask)
                is_overlapping = True
                file.write(f"Overlapping with the [{i}] object\n")
                forget_timer[i] = forget_time
            
        # Forget loop
        nbr_of_forget = 0
        for i in range(len(forget_timer)):
            i = i - nbr_of_forget
            forget_timer[i] = forget_timer[i]-1
            if forget_timer[i] <= 0:
                memorised_labels.pop(i)
                memorised_objects.pop(i)
                memorised_kernels.pop(i)
                forget_timer.pop(i)
                file.write(f"Forget the [{nbr_of_forget+i}] object\n")
                nbr_of_forget += 1

        # ---- create last frame ---- #

        if is_overlapping:

            # Small dilatation to take the cloak edges 
            s = 3
            SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(s,s))
            cloak_mask = cv2.dilate(cloak_mask, SE)

            # What is visible is what is not invisible
            visible_environment_mask = cv2.bitwise_not(cloak_mask)
            visible_environment = cv2.bitwise_and(frame_bgr, frame_bgr, mask=visible_environment_mask)

            if with_first_frame:
                invisible_cloak = cv2.bitwise_and(first_frame, first_frame, mask=cloak_mask)
                output_frame = cv2.add(visible_environment, invisible_cloak)
            else:
                # cv2.INPAINT_NS or cv2.INPAINT_TELEA, cv2.INPAINT_TELEA is smoother
                output_frame = cv2.inpaint(visible_environment, cloak_mask, 3, cv2.INPAINT_TELEA)
        else:
            output_frame = frame_bgr
            file.write(f"No overlapping\n")

        # Write the for the memory video
        if len(memorised_objects):

            # Get the frame of the object
            sum_mask = np.sum(memorised_objects, axis=0)
            binary_mask = (sum_mask > 0).astype(np.uint8) * 255
            object_frame = cv2.bitwise_and(frame_bgr, frame_bgr, mask=binary_mask) 

            # Get the mask of the kernels
            sum_mask = np.sum(memorised_kernels, axis=0)
            binary_mask = (sum_mask > 0).astype(np.uint8) * 255

            # Put mask of the kernels in red
            alpha = 0.5
            object_kernels = cv2.addWeighted(object_frame, 1 - alpha, red_overlay, alpha, 0)

            # Merge the two
            memory_frame = np.where(binary_mask[..., None] > 0, object_kernels, object_frame)

            alpha = 0.35
            green_kernels = cv2.addWeighted(memory_frame, 1 - alpha, green_overlay, alpha, 0)
            memory_frame = np.where(green_mask[..., None] > 0, green_kernels, memory_frame)
            
        else:
            memory_frame = np.zeros_like(frame_bgr)

            alpha = 0.35
            green_kernels = cv2.addWeighted(memory_frame, 1 - alpha, green_overlay, alpha, 0)
            memory_frame = np.where(green_mask[..., None] > 0, green_kernels, memory_frame)

        writer.write(output_frame)
        writer_memory.write(memory_frame)

    file.write(f"End trick 1 \n")
    writer_memory.release()

    return