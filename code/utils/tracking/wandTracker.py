import cv2
import numpy as np
from  ...CONFIG import CONFIG

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

