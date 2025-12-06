import cv2
import numpy as np 
from  ...CONFIG import CONFIG
# =====================================================================
# SIMPLE ONLINE KALMAN TRACKER
# =====================================================================
class KalmanBoxTracker:
    def __init__(self, init_box):
        self.kf = cv2.KalmanFilter(8, 4)
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
# UTILITIES
# =====================================================================

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
