import cv2
import numpy as np
import time
from utils.color import detect_color
from utils.ROI import roi


# ======================================================
# 6-STATE CONSTANT-ACCELERATION KALMAN FILTER
# ======================================================
def create_kf_ca(dt=1.0):
    """
    State vector: [x, y, vx, vy, ax, ay]^T
    """
    kf = cv2.KalmanFilter(6, 2)

    # Transition matrix (CA model)
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0, 0.5*dt*dt, 0],
        [0, 1, 0, dt, 0, 0.5*dt*dt],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=np.float32)

    # Measurement: x, y only
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ], dtype=np.float32)

    # Process noise (allows acceleration)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.005

    # Measurement noise (dynamic later)
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5

    # Initial posterior covariance
    kf.errorCovPost = np.eye(6, dtype=np.float32) * 1

    return kf


# ======================================================
# Extract red blobs (ignore too large & too small)
# ======================================================
def get_red_blobs(mask, min_area=10, max_area=900):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x,y,w,h = cv2.boundingRect(c)

        blobs.append({
            "center": (cx,cy),
            "bbox": (x,y,w,h),
            "area": area
        })

    return blobs



# ======================================================
# FULL 6-STATE KF WAND TRACKER
# ======================================================
def wand_tracker():

    INPUT = "../input/group_3/group_3_out.mp4"
    OUTPUT = "../output/group_3_wand_tracker_stable.mp4"

    cap = cv2.VideoCapture(INPUT)
    if not cap.isOpened():
        print("❌ Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / fps
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H)
    )

    # Kalman
    kf = create_kf_ca(dt)
    kalman_ready = False

    last_pos = None
    frame_idx = 0

    print("▶ Running 6-state KF wand tracker...")

    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out = frame.copy()

        # -------------------------------------------------
        # 1. DETECT RED
        # -------------------------------------------------
        mask = detect_color(frame, [255,0,5], [55,55], [255,255], tuning=25)
        mask = roi(mask, 3, 15)

        blobs = get_red_blobs(mask)

        # -------------------------------------------------
        # KF prediction
        # -------------------------------------------------
        pred = kf.predict()
        px, py = pred[0][0], pred[1][0]

        chosen = None

        # -------------------------------------------------
        # 2. BLOB SELECTION
        # -------------------------------------------------
        if len(blobs) > 0:

            # First time: choose median blob
            if not kalman_ready:
                blobs_sorted = sorted(blobs, key=lambda b: b["area"])
                chosen = blobs_sorted[len(blobs_sorted)//2]

                cx, cy = chosen["center"]
                kf.statePost = np.array([[cx],[cy],[0],[0],[0],[0]], dtype=np.float32)
                kalman_ready = True

            else:
                # Pick blob closest to prediction
                best = 1e9
                for b in blobs:
                    cx, cy = b["center"]
                    dist = np.hypot(cx - px, cy - py)
                    if dist < best:
                        best = dist
                        chosen = b

                # -----------------------------------------------
                # 3. Dynamic measurement noise (motion adaptive)
                # -----------------------------------------------
                speed = np.hypot(pred[2][0], pred[3][0])
                kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * max(3, speed/20)

        # -------------------------------------------------
        # 4. UPDATE or ONLY PREDICT
        # -------------------------------------------------
        if chosen is not None:
            cx, cy = chosen["center"]
            measurement = np.array([[cx],[cy]], dtype=np.float32)
            kf.correct(measurement)
            last_pos = (cx,cy)

        else:
            # No detection → use prediction but LIMIT jump
            if last_pos is not None:
                cx, cy = int(px), int(py)

                # Prevent teleporting
                max_jump = 60
                cx = int(np.clip(cx, last_pos[0] - max_jump, last_pos[0] + max_jump))
                cy = int(np.clip(cy, last_pos[1] - max_jump, last_pos[1] + max_jump))

                last_pos = (cx,cy)
                chosen = {"center": (cx,cy), "bbox": (cx-10, cy-10, 20, 20)}
            else:
                continue

        # -------------------------------------------------
        # 5. DRAW
        # -------------------------------------------------
        x,y,w,h = chosen["bbox"]
        cv2.rectangle(out, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.putText(out, "WAND", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        writer.write(out)

        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}", end="\r")

        frame_idx += 1

    cap.release()
    writer.release()

    print("\n🎉 Done:", OUTPUT)
    print(f"⏱ Time: {time.time()-t0:.2f}s")

# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    wand_tracker()