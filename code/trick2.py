import cv2
import numpy as np

from utils.color import detect_color, change_color_mask
from utils.ROI import roi
from utils.geometry import grow_object



# ======================================================
# LOAD TRACKING LOG (output of your Kalman tracker)
# ======================================================
def load_tracking_log(path):
    """
    Returns:
        tracks[frame] = list of objects {id,cx,cy,w,h}
    """
    tracks = {}

    with open(path, "r") as f:
        next(f)  # skip header

        for line in f:
            frame, tid, cx, cy, w, h = line.strip().split(",")
            frame = int(frame)
            tid   = int(tid)
            cx, cy, w, h = map(float, (cx, cy, w, h))

            if frame not in tracks:
                tracks[frame] = []

            tracks[frame].append({
                "id": tid,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h
            })

    return tracks



# ======================================================
# MAIN TRICK FUNCTION
# ======================================================
def trick2(cap: cv2.VideoCapture,
           writer: cv2.VideoWriter,
           nb_frame,
           log_file_path,
           allowed_ids=None,
           frame_offset=0,
           file=None):

    """
    Parameters
    ----------
    cap : cv2.VideoCapture
    writer : cv2.VideoWriter
    nb_frame : number of frames to process
    to_remove : labels to ignore
    read_every_x_frame : (unused now, but kept for compatibility)
    log_file_path : tracking log .txt file
    allowed_ids : list of tracker IDs to use (None = all)
    frame_offset : integer offset if log starts at frame != trick start
    file : log file for writing (debug)
    """

    # ======================================================
    # LOAD TRACKED OBJECTS
    # ======================================================
    tracking_data = load_tracking_log(log_file_path)

    if allowed_ids is not None:
        allowed_ids = set(allowed_ids)

    if file:
        file.write("Start trick2\n")

    print("\nUsing tracking log:", log_file_path)


    # ======================================================
    # INITIAL STATE
    # ======================================================
    switch = 0
    overlap_prev = False

    last_masks = []
    last_labels = []

    bottle_mask = None
    target_labels = ["bottle", "cell_phone"]  # kept but not really used now


    # ======================================================
    # PROCESS FRAMES
    # ======================================================
    for frame_idx in range(nb_frame):

        ok, frame = cap.read()
        if not ok:
            break

        if file:
            file.write(f"Frame {frame_idx}\n")

        print(f"Trick2 progress: {(frame_idx/nb_frame)*100:.2f}%", end="\r")

        output = frame.copy()

        # ------------------------------------------------------
        # COLOR DETECTION
        # ------------------------------------------------------
        mask_red  = detect_color(frame, [255,0,5], [55,55],[255,255], tuning=25)
        roi_red   = roi(mask_red, 3, 20)

        mask_blue = detect_color(frame, [0,0,255], [100,100],[255,255], tuning=25)
        roi_blue  = roi(mask_blue, 10, 20)


        # ------------------------------------------------------
        # LOAD TRACKED OBJECTS FROM LOG FILE
        # ------------------------------------------------------
        last_masks = []
        last_labels = []

        log_frame_number = frame_idx + frame_offset

        if log_frame_number in tracking_data:
            for obj in tracking_data[log_frame_number]:

                # Filter by ID if needed
                if allowed_ids is not None and obj["id"] not in allowed_ids:
                    continue

                cx, cy = obj["cx"], obj["cy"]
                w, h   = obj["w"], obj["h"]

                # Convert center → corner bbox
                x1 = int(cx - w/2)
                y1 = int(cy - h/2)
                x2 = int(cx + w/2)
                y2 = int(cy + h/2)

                # Clip
                H, W = frame.shape[:2]
                x1 = max(0, min(W-1, x1))
                x2 = max(0, min(W-1, x2))
                y1 = max(0, min(H-1, y1))
                y2 = max(0, min(H-1, y2))

                # Build mask
                mask = np.zeros((H, W), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255

                last_masks.append(mask)
                last_labels.append(f"ID_{obj['id']}")


        # ------------------------------------------------------
        # DRAW TRACKED BOUNDING BOXES
        # ------------------------------------------------------
        for mask, label in zip(last_masks, last_labels):
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()

                cv2.rectangle(output, (x_min, y_min), (x_max, y_max),
                              (0,255,0), 2)
                cv2.putText(output, label, (x_min, y_min-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


        # ------------------------------------------------------
        # WAND INTERACTION (RED touching blue)
        # ------------------------------------------------------
        last_masks.append(roi_blue)
        last_labels.append("blue ball")

        global_overlap = np.zeros_like(roi_red, dtype=np.uint8)

        for mask in last_masks:
            overlap = cv2.bitwise_and(mask, roi_red)
            global_overlap = cv2.bitwise_or(global_overlap, overlap)

        overlap_pixels = np.count_nonzero(global_overlap)
        overlap_now = overlap_pixels > 30   # threshold

        if overlap_now and not overlap_prev:
            switch += 1   # NEW event

        overlap_prev = overlap_now
        state = switch >= 1


        # ------------------------------------------------------
        # TRICK LOGIC
        # ------------------------------------------------------
        mask_blue_cleaned = roi(mask_blue, 10, 15)

        if switch == 1:
            output = change_color_mask(frame, [0,0,255],[255,0,0], mask_blue_cleaned)

        elif switch == 2:
            output = change_color_mask(frame, [0,0,255],[0,255,0], mask_blue_cleaned)

        elif switch == 3:
            output = frame

        elif switch == 4:
            # enlarge first detected object
            target_index = None

            for i, lab in enumerate(last_labels):
                if lab.startswith("ID_"):
                    target_index = i
                    break

            if target_index is not None:
                object_mask = last_masks[target_index]
                output = grow_object(frame, object_mask, scale=3.0)
            else:
                output = frame

        else:
            output = frame


        # ------------------------------------------------------
        # DEBUG INFO
        # ------------------------------------------------------
        cv2.putText(output, f"Overlap: {overlap_pixels}", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)
        cv2.putText(output, f"Switch: {switch}", (50,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
        cv2.putText(output, f"State: {state}", (50,130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

        writer.write(output)


    if file:
        file.write("End trick2\n")

    return 1


