import cv2
import numpy as np
import time


# ============================================================================
# 0. CONFIGURATION (edit here when testing on new videos)
# ============================================================================

CONFIG = {

    # --- Strict red detector ---
    # Higher = require stronger red → reduces false positives
    # Lower  = detect more red → useful if lighting is dark
    "red_min_s": 120,
    "red_min_v": 60,
    "median_blur": 5,      # remove tiny noise dots

    # --- Red blob filtering ---
    # Only accept blobs that look like the wand tip
    "blob_min_area": 8,     # ignore noise
    "blob_max_area": 300,   # ignore large red regions (e.g. mushroom)
    "max_blob_width": 45,   # wand tip is small
    "max_blob_height": 45,

    # --- Kalman filter tuning ---
    # process_noise:
    #     higher  = smoother, slower movement
    #     lower   = more reactive, less smooth
    "kf_process_noise": 0.1,

    # measurement_noise:
    #     higher  = trust detection less (more smoothing)
    #     lower   = trust detection more (faster reactions)
    "kf_measurement_noise": 5.0,

    # --- Limit impossible jumps (px/frame) ---
    # Prevents teleporting to mushroom or bottle
    "max_prediction_jump": 40,

    # --- Backward smoothing (after full tracking) ---
    # If a single frame jumps too far → correct it
    "smooth_jump_threshold": 45,
    "smooth_window": 2,
    "max_frame_gap": 12,   # do NOT smooth when wand is out of frame too long

    # --- Output bounding box size (in log file) ---
    "bbox_size": 16
}

# ============================================================================
# LOAD OBJECT LOG (trick2_log)
# ============================================================================

def load_object_log(path):
    """
    Reads: Frame,ID,cx,cy,w,h
    Returns: dict[frame] = list of {cx,cy,w,h}
    """
    objects = {}

    with open(path, "r") as f:
        next(f)
        for line in f:
            frame, tid, cx, cy, w, h = line.strip().split(",")
            frame = int(frame)
            cx, cy, w, h = map(float, (cx, cy, w, h))

            if frame not in objects:
                objects[frame] = []

            objects[frame].append({
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h
            })

    return objects




# ============================================================================
# REMOVE OBJECTS FROM FRAME
# ============================================================================

def remove_objects(frame, object_list):
    """
    Paints object bounding boxes black (removes them).
    So wand-detector sees a clean frame.
    """
    if object_list is None:
        return frame

    clean = frame.copy()
    H, W = frame.shape[:2]

    for obj in object_list:
        cx, cy, w, h = obj["cx"], obj["cy"], obj["w"], obj["h"]

        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)

        x1 = max(0, min(W-1, x1))
        x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1))
        y2 = max(0, min(H-1, y2))

        clean[y1:y2, x1:x2] = 0

    return clean




# ============================================================================
# 1. STRICT RED DETECTION
# ============================================================================

def detect_red_strict(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, CONFIG["red_min_s"], CONFIG["red_min_v"]])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, CONFIG["red_min_s"], CONFIG["red_min_v"]])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.medianBlur(mask, CONFIG["median_blur"])
    return mask



# ============================================================================
# 2. RED BLOBS EXTRACTION
# ============================================================================

def extract_red_blobs(mask, cfg):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []

    for c in contours:
        area = cv2.contourArea(c)
        if not (cfg["blob_min_area"] <= area <= cfg["blob_max_area"]):
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w > cfg["max_blob_width"] or h > cfg["max_blob_height"]:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])

        blobs.append({"center": (cx,cy), "bbox": (x,y,w,h), "area": area})

    return blobs



# ============================================================================
# 3. KALMAN FILTER
# ============================================================================

def create_kalman(cfg):
    kf = cv2.KalmanFilter(4, 2)
    dt = 1.0

    kf.transitionMatrix = np.array([
        [1,0,dt,0],
        [0,1,0,dt],
        [0,0,1,0],
        [0,0,0,1]
    ], dtype=np.float32)

    kf.measurementMatrix = np.eye(2,4, dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * cfg["kf_process_noise"]
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * cfg["kf_measurement_noise"]

    return kf



# ============================================================================
# 4. BACKWARD SMOOTHING
# ============================================================================

def backward_smooth(track, cfg):
    if len(track) < 3:
        return track

    out = track.copy()

    for i in range(1, len(out)-1):
        p_prev = np.array([out[i-1]["cx"], out[i-1]["cy"]])
        p_curr = np.array([out[i]["cx"], out[i]["cy"]])
        p_next = np.array([out[i+1]["cx"], out[i+1]["cy"]])

        if np.linalg.norm(p_curr - p_prev) > cfg["smooth_jump_threshold"] and \
           np.linalg.norm(p_curr - p_next) > cfg["smooth_jump_threshold"]:

            if (out[i]["frame"] - out[i-1]["frame"]) > cfg["max_frame_gap"]:
                continue
            if (out[i+1]["frame"] - out[i]["frame"]) > cfg["max_frame_gap"]:
                continue

            new_pos = (p_prev + p_next) / 2
            out[i]["cx"] = int(new_pos[0])
            out[i]["cy"] = int(new_pos[1])

    return out



# ============================================================================
# 5. MAIN TRACKER (YOUR ORIGINAL CODE + OBJECT REMOVAL)
# ============================================================================

def wand_tracker(
    video_path,
    removal_log_path,     # NEW: trick2_log used to clean the frame
    output_video,
    output_txt,
    start_frame=0,
    cfg=CONFIG
):

    # Load object-removal log
    obj_log = load_object_log(removal_log_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ ERROR: Cannot open input video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    log = open(output_txt, "w")
    log.write("frame,id,cx,cy,w,h\n")

    kf = create_kalman(cfg)
    kalman_ready = False
    last_pos = None

    results = []
    frame_id = 0

    print("▶ Tracking wand...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_id < start_frame:
            writer.write(frame)
            frame_id += 1
            continue

        # ------------------------------------------------------
        # REMOVE ALL TRACKED OBJECTS BEFORE WAND DETECTION
        # ------------------------------------------------------
        objects_here = obj_log.get(frame_id, None)
        cleaned = remove_objects(frame, objects_here)

        # detect wand in cleaned frame
        mask = detect_red_strict(cleaned)
        blobs = extract_red_blobs(mask, cfg)

        # prediction
        pred = kf.predict()
        px, py = float(pred[0]), float(pred[1])

        if last_pos is not None:
            px = np.clip(px, last_pos[0]-cfg["max_prediction_jump"],
                              last_pos[0]+cfg["max_prediction_jump"])
            py = np.clip(py, last_pos[1]-cfg["max_prediction_jump"],
                              last_pos[1]+cfg["max_prediction_jump"])

        chosen = None

        if blobs:
            if not kalman_ready:
                blobs_sorted = sorted(blobs, key=lambda b:b["area"])
                chosen = blobs_sorted[len(blobs_sorted)//2]
                cx,cy = chosen["center"]
                kf.statePost = np.array([[cx],[cy],[0],[0]], dtype=np.float32)
                kalman_ready = True
            else:
                best = 999999
                for b in blobs:
                    cx,cy = b["center"]
                    d = np.hypot(cx-px, cy-py)
                    if d < best:
                        best = d
                        chosen = b

        if chosen is None:
            cx, cy = int(px), int(py)
            x = cx - cfg["bbox_size"]//2
            y = cy - cfg["bbox_size"]//2
            w = h = cfg["bbox_size"]
        else:
            cx,cy = chosen["center"]
            x,y,w,h = chosen["bbox"]
            kf.correct(np.array([[cx],[cy]], dtype=np.float32))

        last_pos = (cx,cy)

        # draw wand on ORIGINAL FRAME
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),2)
        cv2.putText(frame, "WAND", (x,y-8), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        writer.write(frame)

        log.write(f"{frame_id},100,{cx},{cy},{w},{h}\n")
        results.append({"frame":frame_id, "cx":cx, "cy":cy, "w":w, "h":h})

        frame_id += 1

    cap.release()
    writer.release()
    log.close()

    print("▶ Forward pass complete — running smoothing...")

    smoothed = backward_smooth(results, cfg)

    smoothed_txt = output_txt.replace(".txt", "_smoothed.txt")
    with open(smoothed_txt, "w") as f:
        f.write("frame,id,cx,cy,w,h\n")
        for s in smoothed:
            f.write(f"{s['frame']},100,{s['cx']},{s['cy']},{s['w']},{s['h']}\n")
    
    print("▶ Generating smoothed video...")

    # --- Re-open video ---
    cap2 = cv2.VideoCapture(video_path)
    smoothed_video = output_video.replace(".mp4", "_smoothed.mp4")

    writer2 = cv2.VideoWriter(
        smoothed_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    # Prepare dict: frame → (cx,cy,w,h)
    smooth_dict = { t["frame"]: t for t in smoothed }

    fid = 0
    while True:
        ok, frame2 = cap2.read()
        if not ok:
            break

        if fid in smooth_dict:
            s = smooth_dict[fid]
            cx, cy, w, h = s["cx"], s["cy"], s["w"], s["h"]
            x = cx - w//2
            y = cy - h//2

            cv2.rectangle(frame2, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame2, "WAND (smoothed)", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        writer2.write(frame2)
        fid += 1

    cap2.release()
    writer2.release()

    print("🎉 Smoothed video saved at:", smoothed_video)

    print("🎉 DONE!")
    print("➡ Video:", output_video)
    print("➡ Track:", output_txt)
    print("➡ Smoothed:", smoothed_txt)



# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    wand_tracker(
        "../input/dynamic/trick2.mp4",
        "../output/trick2_log.txt",            # <--- REMOVE these objects first
        "../output/wand_tracker_final.mp4",
        "../output/wand_tracker_final.txt",
        start_frame=150,
        cfg=CONFIG
    )
