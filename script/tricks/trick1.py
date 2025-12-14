import cv2
import os
import numpy as np

from script.CONFIG import *
from script.utils.detection.color import detect_green
from script.utils.mask_operation import get_overlap_componant, roi
from script.utils.tracking.trajectory import object_trajectory

def create_kernel(mask, dim, kernel_fraction):
    """
    Create the kernel mask of the given object

    Args:
        mask: The mask of the object
        dim: The dimention (w,h) of the bounding box of the object
        kernel_fraction: The fraction of the object that is 
                        remove to create the kernel (how much it is "shrink")

    Return:
        The mask of the kernel
    """
    w, h = max(1,int(dim[0]*kernel_fraction)), max(1,int(dim[1]*kernel_fraction))
    kernel_er = [w, h]
    adaptive_SE = cv2.getStructuringElement(cv2.MORPH_RECT,kernel_er)
    kernel = cv2.erode(mask, adaptive_SE)
    return kernel

def trick1(writer, cap, nb_frame, traj_path, debug=True):
    """
    Write the next nbr_frame in the cap, with the trick 1 effect.

    Args:
        writer : writer of the output video
        cap : the cap of the video
        traj_path : the object trajectory .txt
        debug : Bool to make the debugging video or not
    """

    log_path = f"files/trick1/{FILE_NAME}.txt"
    obj_video_path = f"files/trick1/{FILE_NAME}.mp4"

    log = open(log_path, "w", encoding="utf-8")
    obj_trajs = object_trajectory(traj_path)

    log.write(f"Start trick 1 \n")
    log.write(f"With first frame = {WITH_FIRST_FRAME} \n")
    log.write(f"Trajectory file : {traj_path} \n")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_debug = cv2.VideoWriter(obj_video_path, fourcc, fps, (w, h))

    ok, first_frame = cap.read()
    if not ok:
        print("no frames")

    # Use to show object kernels in debugging video
    red_overlay = np.zeros_like(first_frame)
    red_overlay[:, :] = (0, 0, 255)

    # Use to show green object in debugging video
    green_overlay = np.zeros_like(first_frame)
    green_overlay[:, :] = (0, 255, 0)

    # kernel = center on the object
    object_kernels = []

    log.write(f"Adaptative fraction of s_erosion for kernels = {KERNEL_FRACTION} \n")

    for frame_idx in range(nb_frame):

        ok, frame = cap.read()
        if not ok:
            break

        print(f"Trick 1 progress: {(frame_idx)/(nb_frame) *100:.2f} %", end="\r")
        log.write(f"\n   Trick 1 at {frame_idx/fps:.3f} s ({(frame_idx)/(nb_frame) *100:.2f} %)\n")

        # Get the green mask
        green_mask = detect_green(frame)

        # Filter the parasitic green pixels
        green_mask = roi(green_mask, S_ERODE_ROI, S_DIL_ROI)

        # Filter the parasitic green pixels
        cloak_mask = np.zeros_like(frame[:,:,1])
        is_overlapping = False
        object_kernels.clear()

        # ---- Get the objects ---- #

        obj_masks = obj_trajs.masks_at_frame(frame_idx, frame)
        obj_boxes = obj_trajs.boxs_at_frame(frame_idx)

        valid_masks = [mask for mask in obj_masks if mask is not None]
        valid_boxes = [box for box in obj_boxes if box is not None]

        log.write(f"Number of valid boxes {len(valid_boxes)}\n")
        log.write(f"Object boxes {obj_boxes}\n")

        # ---- Get the corresponding kernels ---- #

        for mask, box in zip(valid_masks, valid_boxes):

            _, _, w, h = box
            object_kernels.append(create_kernel(mask, (w,h), KERNEL_FRACTION))

        # ---- Look at overlap on the kernels ---- #

        for i, kernel in enumerate(object_kernels):

            overlapping, overlap_mask = get_overlap_componant(green_mask, kernel)

            if overlapping:
                cloak_mask = cv2.add(cloak_mask, overlap_mask)
                is_overlapping = True
                log.write(f"Overlapping with kernel [{i}]\n")

        # ---- create new frame ---- #

        if is_overlapping:

            # Small dilatation to take the cloak edges 
            SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(S_DIL_CLOAK,S_DIL_CLOAK))
            cloak_mask = cv2.dilate(cloak_mask, SE)

            # What is visible is what is not invisible
            visible_environment_mask = cv2.bitwise_not(cloak_mask)
            visible_environment = cv2.bitwise_and(frame, frame, mask=visible_environment_mask)

            # Type of invisibility
            if WITH_FIRST_FRAME:
                invisible_cloak = cv2.bitwise_and(first_frame, first_frame, mask=cloak_mask)
                output_frame = cv2.add(visible_environment, invisible_cloak)
            else:
                # cv2.INPAINT_NS or cv2.INPAINT_TELEA, cv2.INPAINT_TELEA is smoother
                output_frame = cv2.inpaint(visible_environment, cloak_mask, 3, cv2.INPAINT_TELEA)
        else:
            output_frame = frame
            log.write(f"No overlapping\n")

        # ---- Write for the object video ---- #

        if debug:
            if len(valid_masks):

                # Get the frame of the object
                sum_mask = np.sum(valid_masks, axis=0)
                binary_mask = (sum_mask > 0).astype(np.uint8) * 255
                object_frame = cv2.bitwise_and(frame, frame, mask=binary_mask) 

                # Get the mask of the kernels
                sum_mask = np.sum(object_kernels, axis=0)
                binary_mask = (sum_mask > 0).astype(np.uint8) * 255

                # Put mask of the kernels in red
                alpha = 0.5
                obj_kernels = cv2.addWeighted(object_frame, 1 - alpha, red_overlay, alpha, 0)

                # Merge the two
                obj_frame = np.where(binary_mask[..., None] > 0, obj_kernels, object_frame)

                alpha = 0.35
                green_kernels = cv2.addWeighted(obj_frame, 1 - alpha, green_overlay, alpha, 0)
                obj_frame = np.where(green_mask[..., None] > 0, green_kernels, obj_frame)

            else:
                obj_frame = np.zeros_like(frame)

                alpha = 0.35
                green_kernels = cv2.addWeighted(obj_frame, 1 - alpha, green_overlay, alpha, 0)
                obj_frame = np.where(green_mask[..., None] > 0, green_kernels, obj_frame)

            writer_debug.write(obj_frame)

        writer.write(output_frame)

    log.write(f"End trick 1 \n")
    writer_debug.release()

    return