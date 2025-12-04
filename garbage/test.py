import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import os
# =====================================================================
# CONFIGURATION
# =====================================================================
VIDEO_IN = "../input/dynamic/trick1.mp4"
VIDEO_OUT_FORWARD = "../output/trick1_forward.mp4"
VIDEO_OUT_RTS = "../output/trick1_rts.mp4"
LOG_PATH = "../output/tric1_log.txt"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

YOLO_MODEL = "yolov8n.pt"
CONF_THR = 0.25
IOU_THR = 0.20
N_OBJECTS = 3   # number of objects to click and track


# =====================================================================
# UTILS
# =====================================================================
def xyxy_to_cxcywh(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return np.array([cx, cy, w, h], dtype=np.float32)

def cxcywh_to_xyxy(state):
    cx, cy, w, h = state[:4]
    return int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)

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
# KALMAN TRACKER (cx,cy,w,h + velocities)
# =====================================================================
class KalmanBoxTracker:
    def __init__(self, init_bbox):
        """
        init_bbox = (cx,cy,w,h)
        """
        self.kf = cv2.KalmanFilter(8, 4)
        dt = 1.0

        # Transition matrix
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
        self.kf.processNoiseCov   = np.eye(8, dtype=np.float32) * 0.001
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.05

        cx,cy,w,h = init_bbox
        self.kf.statePost = np.array([[cx],[cy],[w],[h],[0],[0],[0],[0]], dtype=np.float32)

        self.history = []
        self.P_history = []
        self.pred_history = []
        self.Ppred_history = []

    def predict(self):
        pred = self.kf.predict()
        self.pred_history.append(pred.copy())
        self.Ppred_history.append(self.kf.errorCovPre.copy())
        return pred[:4].ravel()

    def update(self, det):
        meas = np.array(det, dtype=np.float32).reshape((4,1))
        self.kf.correct(meas)

    def record(self):
        self.history.append(self.kf.statePost.copy())
        self.P_history.append(self.kf.errorCovPost.copy())


# =====================================================================
# RTS SMOOTHER
# =====================================================================
def rts_smoother(trk):
    F = trk.kf.transitionMatrix
    Q = trk.kf.processNoiseCov

    Xf = trk.history
    Pf = trk.P_history
    Xp = trk.pred_history
    Pp = trk.Ppred_history

    T = len(Xf)
    Xs = [None] * T
    Ps = [None] * T

    Xs[-1] = Xf[-1].copy()
    Ps[-1] = Pf[-1].copy()

    for t in reversed(range(T-1)):
        C = Pf[t] @ F.T @ np.linalg.inv(Pp[t])
        Xs[t] = Xf[t] + C @ (Xs[t+1] - Xp[t])
        Ps[t] = Pf[t] + C @ (Ps[t+1] - Pp[t]) @ C.T

    return Xs


# =====================================================================
# MOUSE CLICK UI
# =====================================================================
clicks = []
current_frame = None

def mouse_callback(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print("Clicked:", x, y)


def select_bboxes(frame):

    global clicks, current_frame
    clicks = []
    current_frame = frame.copy()

    cv2.namedWindow("Select Objects")
    cv2.setMouseCallback("Select Objects", mouse_callback)

    print(f"\n🔵 Click TOP-LEFT then BOTTOM-RIGHT for {N_OBJECTS} objects")

    while True:
        display = current_frame.copy()

        for i in range(0, len(clicks), 2):
            if i+1 < len(clicks):
                cv2.rectangle(display, clicks[i], clicks[i+1], (0,255,0), 2)

        cv2.imshow("Select Objects", display)
        if cv2.waitKey(20) == 27:
            break

        if len(clicks) == N_OBJECTS * 2:
            break

    cv2.destroyWindow("Select Objects")

    # Convert clicks to bounding boxes
    init_bboxes = []
    for i in range(0, N_OBJECTS * 2, 2):
        (x1,y1), (x2,y2) = clicks[i], clicks[i+1]
        init_bboxes.append(xyxy_to_cxcywh(x1,y1,x2,y2))

    return init_bboxes


# =====================================================================
# MAIN
# =====================================================================
def main():

    # Ask user for starting frame
    start_frame = int(input("Enter frame number where objects appear: "))

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise IOError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Jump to that frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {start_frame}")

    # Let user click objects
    init_bboxes = select_bboxes(frame0)

    # Create Kalman trackers
    trackers = [KalmanBoxTracker(b) for b in init_bboxes]

    # YOLO model
    model = YOLO(YOLO_MODEL)

    # Writers
    writer_forward = cv2.VideoWriter(VIDEO_OUT_FORWARD,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (W,H))

    log = open(LOG_PATH, "w")
    log.write("Frame,ID,cx,cy,w,h\n")

    # ==============================================================
    # FORWARD PASS
    # ==============================================================
    print("\n▶ Forward tracking...\n")

    frame_idx = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        ok, frame = cap.read()
        if not ok: break

        # YOLO detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb, verbose=False)[0]

        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THR:
                continue

            x1,y1,x2,y2 = box.xyxy[0]
            detections.append(xyxy_to_cxcywh(float(x1), float(y1), float(x2), float(y2)))

        # Track update
        for tid, trk in enumerate(trackers):
            pred = trk.predict()
            pred_bb = cxcywh_to_xyxy(pred)

            best_det = None
            best_i = 0

            for det in detections:
                i = iou(pred_bb, cxcywh_to_xyxy(det))
                if i > best_i:
                    best_i = i
                    best_det = det

            if best_i > IOU_THR:
                trk.update(best_det)

            trk.record()

        # Draw forward result
        vis = frame.copy()
        for tid, trk in enumerate(trackers):
            state = trk.kf.statePost[:4].ravel()
            x1,y1,x2,y2 = cxcywh_to_xyxy(state)
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f"ID {tid}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cx,cy,w,h = state
            log.write(f"{frame_idx},{tid},{cx},{cy},{w},{h}\n")

        writer_forward.write(vis)
        print(f"\rFrame {frame_idx}/{total}", end="")
        frame_idx += 1

    cap.release()
    writer_forward.release()
    log.close()

    print("\nForward tracking done.")

    # ==============================================================
    # RTS SMOOTHING
    # ==============================================================
    print("▶ Running RTS smoothing...")
    smoothed_tracks = [rts_smoother(trk) for trk in trackers]
    print("RTS done.")

    # ==============================================================
    # RENDER SMOOTHED VIDEO
    # ==============================================================
    print("▶ Rendering smoothed video...")

    cap = cv2.VideoCapture(VIDEO_IN)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer_rts = cv2.VideoWriter(VIDEO_OUT_RTS,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (W,H))

    frame_idx = start_frame
    while True:
        ok, frame = cap.read()
        if not ok: break

        f = frame.copy()
        s_idx = frame_idx - start_frame

        for tid in range(N_OBJECTS):
            if s_idx < len(smoothed_tracks[tid]):
                state = smoothed_tracks[tid][s_idx][:4].ravel()
                x1,y1,x2,y2 = cxcywh_to_xyxy(state)

                cv2.rectangle(f,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.putText(f,f"ID {tid}",(x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

        writer_rts.write(f)
        print(f"\rFrame {frame_idx}/{total}", end="")
        frame_idx += 1

    writer_rts.release()
    cap.release()

    print("\nAll done.")
 

# =====================================================================
# RUN
# =====================================================================
if __name__ == "__main__":
    main()
