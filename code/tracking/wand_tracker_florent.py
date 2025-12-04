import cv2
import numpy as np


# CONFIG — calibré pour TA baguette (HEX f53145)


CONFIG = {
    "median_blur": 5,
    "blob_min_area": 10,
    "blob_max_area": 250,
    "max_blob_width": 45,
    "max_blob_height": 45,
    "kf_process_noise": 0.1,
    "kf_measurement_noise": 5.0,
    "max_prediction_jump": 35,
    "bbox_size": 14
}

def detect_red_strict(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Wand tip (#f53145) : hue ≈ 177
    lower = np.array([165, 150, 150], dtype=np.uint8)
    upper = np.array([185, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.GaussianBlur(mask, (5,5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


# BLOBS extraction — seulement dans le bas de l’image

def extract_red_blobs(mask, cfg):
    H = mask.shape[0]

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []

    for c in contours:
        area = cv2.contourArea(c)
        if not (cfg["blob_min_area"] <= area <= cfg["blob_max_area"]):
            continue

        x, y, w, h = cv2.boundingRect(c)

        # important : la baguette est tenue → jamais en haut
        if y < H * 0.25:
            continue

        if w > cfg["max_blob_width"] or h > cfg["max_blob_height"]:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        blobs.append({"center": (cx, cy), "bbox": (x, y, w, h), "area": area})

    return blobs

# Kalman

def create_kalman(cfg):
    kf = cv2.KalmanFilter(4, 2)
    dt = 1.0

    kf.transitionMatrix = np.array([
        [1,0,dt,0],
        [0,1,0,dt],
        [0,0,1,0],
        [0,0,0,1]
    ], dtype=np.float32)

    kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * cfg["kf_process_noise"]
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * cfg["kf_measurement_noise"]

    return kf

# TRACKER

class WandTracker:

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        self.kf = create_kalman(cfg)
        self.ready = False

        self.last = None
        self.last_vx = 0
        self.last_vy = 0

    def update(self, frame):

        mask = detect_red_strict(frame)
        blobs = extract_red_blobs(mask, self.cfg)

        # ---- PREDICTION ----
        pred = self.kf.predict()
        px, py, vx, vy = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])

        # direction estimation
        self.last_vx = 0.8*self.last_vx + 0.2*vx
        self.last_vy = 0.8*self.last_vy + 0.2*vy

        chosen = None
        best_score = 1e9

        # ---- SELECT BLOB ----
        for b in blobs:
            cx, cy = b["center"]

            # 1. distance score
            dist = np.hypot(cx - px, cy - py)

            # 2. directional score
            dx = cx - px
            dy = cy - py
            dir_score = 0

            if abs(self.last_vx) > 1:
                if np.sign(dx) != np.sign(self.last_vx):
                    dir_score += 40         # incompatible horizontal direction

            if abs(self.last_vy) > 1:
                if np.sign(dy) != np.sign(self.last_vy):
                    dir_score += 40         # incompatible vertical direction

            total = dist + dir_score

            if total < best_score:
                best_score = total
                chosen = b

        # ---- UPDATE KF ----
        if chosen is not None and best_score < 80:
            cx, cy = chosen["center"]
            x,y,w,h = chosen["bbox"]
            self.kf.correct(np.array([[cx],[cy]], dtype=np.float32))
        else:
            # no detection → use pure prediction
            cx, cy = int(px), int(py)
            w = h = self.cfg["bbox_size"]
            x = cx - w//2
            y = cy - h//2

        self.last = (cx,cy)
        return (cx, cy, x, y, w, h)
