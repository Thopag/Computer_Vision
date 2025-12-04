import cv2
import numpy as np
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =====================================================================
# CONFIGURATION
# =====================================================================
CONFIG = {
    "IN_PATH": "../input/video_group_11_dynamic.mp4",
    "OUT_PATH": "../output/trackerV2_test.mp4",
    "LOG_PATH": "../output/trackerV2_test.txt",

    "CONF_THR": 0.25,
    "IOU_THR": -np.inf,
    "MODEL": "yolov8n.pt",
    "N_OBJECTS": 3,
    "nbr_frame_before_sleep": 30,
    "blacklist" : ['person', 'skateboard', 'laptop', 'cup', 'chair', 'dining table', 'microwave', 'umbrella', 
    'kite', 'cat', 'traffic light', 'book', 'cell phone', 'keyboard', 'scissors', 'frisbee', 'suitcase', 'dog', 'tv']

}

# =====================================================================
# UTILITIES
# ============================================2
#========================

def xyxy_to_cxcywh(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2
    return np.array([cx, cy, w, h], dtype=np.float32)

def cxcywh_to_xyxy(state):
    cx, cy, w, h = state[:4]
    return int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)

def diou(bb1, bb2):
    x1, y1, x2, y2 = bb1
    xx1, yy1, xx2, yy2 = bb2

    # --- IoU ---
    xi1 = max(x1, xx1)
    yi1 = max(y1, yy1)
    xi2 = min(x2, xx2)
    yi2 = min(y2, yy2)

    w_inter = max(0, xi2 - xi1)
    h_inter = max(0, yi2 - yi1)
    inter = w_inter * h_inter

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xx2 - xx1) * (yy2 - yy1)
    union = area1 + area2 - inter + 1e-6
    iou = inter / union

    # --- Center distance penalty ---
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    cx2 = (xx1 + xx2) / 2
    cy2 = (yy1 + yy2) / 2

    center_dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2

    # diagonal length of minimum enclosing box
    enc_x1 = min(x1, xx1)
    enc_y1 = min(y1, yy1)
    enc_x2 = max(x2, xx2)
    enc_y2 = max(y2, yy2)

    enc_diag_sq = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2 + 1e-6

    # --- DIoU score ---
    diou_score = iou - (center_dist_sq / enc_diag_sq)
    return diou_score

def iou(bb1, bb2):
    x1,y1,x2,y2 = bb1
    xx1,yy1,xx2,yy2 = bb2

    xi1 = max(x1, xx1)
    yi1 = max(y1, yy1)
    xi2 = min(x2, xx2)
    yi2 = min(y2, yy2)

    w = max(0, xi2 - xi1)
    h = max(0, yi2 - yi1)
    inter = w*h

    area1 = (x2-x1)*(y2-y1)
    area2 = (xx2-xx1)*(yy2-yy1)

    return inter / (area1 + area2 - inter + 1e-6)

# =====================================================================
# SIMPLE ONLINE KALMAN TRACKER
# =====================================================================
class KalmanBoxTracker:
    def __init__(self, init_box, time_before_sleep):
        self.kf = cv2.KalmanFilter(8, 4)
        self.time_before_sleep = time_before_sleep
        self.timer = time_before_sleep

        dt = 1.0

        # State transition: [cx cy w h vx vy vw vh]
        self.kf.transitionMatrix = np.array([
            [1,0,0,0, dt,0,0,0],
            [0,1,0,0, 0,dt,0,0],
            [0,0,1,0, 0,0,dt,0],
            [0,0,0,1, 0,0,0,dt],
            [0,0,0,0, 1,0,0,0],
            [0,0,0,0, 0,1,0,0],
            [0,0,0,0, 0,0,1,0],
            [0,0,0,0, 0,0,0,1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.001
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.05

        cx,cy,w,h = init_box
        self.kf.statePost = np.array([[cx],[cy],[w],[h],[0],[0],[0],[0]], dtype=np.float32)

    def predict(self):
        return self.kf.predict()[:4].ravel()

    def update(self, det):
        meas = np.array(det, dtype=np.float32).reshape(4,1)
        self.kf.correct(meas)

# =====================================================================
# MOUSE CLICK SELECTION
# =====================================================================
clicks = []
current_frame = None

def mouse_callback(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x,y))
        print("Clicked:", (x,y))

def select_bboxes(frame):
    global clicks, current_frame
    clicks = []
    current_frame = frame.copy()

    cv2.namedWindow("Select Objects", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Objects", mouse_callback)

    print(f"\nSelect {CONFIG['N_OBJECTS']} objects: click TOP-LEFT then BOTTOM-RIGHT")

    while True:
        temp = current_frame.copy()
        for i in range(0, len(clicks), 2):
            if i+1 < len(clicks):
                cv2.rectangle(temp, clicks[i], clicks[i+1], (0,255,0), 2)
        cv2.imshow("Select Objects", temp)

        if len(clicks) == CONFIG["N_OBJECTS"]*2:
            break
        if cv2.waitKey(20) == 27:
            break

    cv2.destroyWindow("Select Objects")

    boxes = []
    for i in range(0, len(clicks), 2):
        (x1,y1),(x2,y2) = clicks[i], clicks[i+1]
        boxes.append(xyxy_to_cxcywh(x1,y1,x2,y2))
    return boxes


# =====================================================================
# MAIN
# =====================================================================
def main():
    try:
        start_frame = int(input("Enter frame number where objects appear: "))

        cap = cv2.VideoCapture(CONFIG["IN_PATH"])
        if not cap.isOpened():
            raise IOError("Cannot open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # jump to selection frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame0 = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {start_frame}")

        # SELECT OBJECTS
        init_boxes = select_bboxes(frame0)

        # CREATE TRACKERS
        trackers = [KalmanBoxTracker(b) for b in init_boxes]
        trackers_timers = np.array([CONFIG["nbr_frame_before_sleep"]] * CONFIG["N_OBJECTS"])
        trackers_last_pred = {}

        # YOLO MODEL
        model = YOLO(CONFIG["MODEL"])

        # OUTPUT VIDEO
        writer = cv2.VideoWriter(CONFIG["OUT_PATH"],
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (W,H))

        log = open(CONFIG["LOG_PATH"], "w")
        log.write("Frame,ID,cx,cy,w,h, ACTIVE(timer)\n")

        print("\n▶ Online Kalman tracking...")

        # reset to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # YOLO detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb, verbose=False)[0]
            labels = results.names
            detections = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                label = labels[class_id]
                if label not in CONFIG["blacklist"]:
                    if float(box.conf[0]) < CONFIG["CONF_THR"]:
                        continue
                    x1,y1,x2,y2 = box.xyxy[0]
                    detections.append(xyxy_to_cxcywh(float(x1),float(y1),float(x2),float(y2)))

            if len(detections) > 0:
                all_i_per_det = np.zeros( (len(detections), CONFIG["N_OBJECTS"]), dtype=float)
                for tid, trk in enumerate(trackers):
                    trackers_timers[tid] = trackers_timers[tid] - 1
                    if trackers_timers[tid] > 0:
                        pred = trk.predict()
                        trackers_last_pred[tid] = cxcywh_to_xyxy(pred)

                    for deti, det in enumerate(detections):
                        i = diou(trackers_last_pred[tid], cxcywh_to_xyxy(det))
                        all_i_per_det[deti][tid] = i

                best_matches = []
                for i_vec, det in zip(all_i_per_det, detections):
                    best_match = np.argmax(i_vec)
                    if best_match not in best_matches:
                        trackers[best_match].update(det)
                        trackers_timers[best_match] = CONFIG["nbr_frame_before_sleep"]
                    best_matches.append(best_match)

            # DRAW
            vis = frame.copy()
            for tid, trk in enumerate(trackers):
                
                if trackers_timers[tid] > 0:
                    cx,cy,w,h = trk.kf.statePost[:4].ravel()
                    log.write(f"{frame_idx},{tid},{cx},{cy},{w},{h}, ACTIVE({trackers_timers[tid]})\n")

                    state = trk.kf.statePost[:4].ravel()
                    x1,y1,x2,y2 = cxcywh_to_xyxy(state)

                    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(vis,f"ID {tid}",(x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                else:
                    state = trk.kf.statePost[:4].ravel()
                    x1,y1,x2,y2 = cxcywh_to_xyxy(state)

                    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.putText(vis,f"ID {tid}",(x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)


            writer.write(vis)
            print(f"\rFrame {frame_idx}/{total}", end="")
            frame_idx += 1

        cap.release()
        writer.release()
        log.close()
        print("\nDONE. Tracking video saved:", CONFIG["OUT_PATH"])

    except KeyboardInterrupt:
        print("\n🛑 CTRL+C pressed")

    finally:
        try: cap.release()
        except: pass
        try: writer.release()
        except: pass
        cv2.destroyAllWindows()
        print("🔻 Clean exit")


# =====================================================================
# RUN
# =====================================================================
if __name__ == "__main__":
    main()
