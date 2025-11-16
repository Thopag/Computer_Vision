import cv2
import numpy as np

from utils.color import detect_color
from utils.ROI import get_overlap_componant, detect_objects, is_some_overlap

def trick1(writer, cap, nbr_frame, blacklist=[], with_first_frame=False):
    """
    Writer the next nbr_frame in the cap, with thhe trick 1.
    Blacklist are the detected object that should be ignored.

    writer : writer of the output video
    cap : the cap of the video
    """

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    writer_memory = cv2.VideoWriter("../outputs/trick1_memory.mp4", fourcc, fps, (w, h))

    ok, first_frame = cap.read()
    if not ok:
        print("no frames")

    # Use to show object kernels in memory video
    red_overlay = np.zeros_like(first_frame)
    red_overlay[:, :] = (0, 0, 255)

    # kernel = center on the object
    memorised_objects = []
    memorised_labels = []
    memorised_kernels = []

    s_erode = 17
    kernel_er = (s_erode , s_erode)
    SE_object = cv2.getStructuringElement(cv2.MORPH_RECT,kernel_er)

    for i in range(nbr_frame):

        ok, frame_bgr = cap.read()
        if not ok:
            break

        print(f"Trick 1 progress: {(i)/(nbr_frame) *100:.2f} %. Memorised labels {memorised_labels}", end="\r")

        # Get the green mask
        green = [0,255,0]
        green_mask = detect_color(frame_bgr, green, [40,40], [255,255], tuning=35)

        cloak_mask = np.zeros_like(frame_bgr[:,:,1])
        is_overlapping = False

        mask_objects, labels = detect_objects(frame_bgr)

        # Update memory loop
        for label, mask_object in zip(labels, mask_objects):

            if label not in blacklist:

                in_memory = False
                for i in range(len(memorised_objects)):

                    # Verif if the current object is already in the memory and if yes, update it
                    # We consider it is the same object if at least 1 pixel overlap, to take account of mouvement
                    if is_some_overlap(memorised_objects[i], mask_object):
                        memorised_labels[i] = label
                        memorised_objects[i] = mask_object
                        memorised_kernels[i] = cv2.erode(mask_object, SE_object)
                        in_memory = True

                if not in_memory:
                    memorised_labels.append(label)
                    memorised_objects.append(mask_object)
                    memorised_kernels.append(cv2.erode(mask_object, SE_object))

        # Search overlapping loop
        for memorised_kernel in memorised_kernels:

            # Check if overlap
            overlapping, overlap_mask = get_overlap_componant(green_mask, memorised_kernel)

            # Add the componant into the cloak (invisibility part)
            if overlapping:
                cloak_mask = cv2.add(cloak_mask, overlap_mask)
                is_overlapping = True

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
        else:
            memory_frame = np.zeros_like(frame_bgr)

        writer.write(output_frame)
        writer_memory.write(memory_frame)

    writer_memory.release()

    return