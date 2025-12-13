import numpy as np
import math
import cv2

from script.CONFIG import ALPHA_CIOU

def xyxy_to_cxcywh(state):
    x1, y1, x2, y2 = state
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2
    return np.array([cx, cy, w, h], dtype=np.float32)

def cxcywh_to_xyxy(state):
    cx, cy, w, h = state
    return int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)

def ciou(bb1, bb2):
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

    w1, h1 = x2 - x1, y2 - y1
    w2, h2 = xx2 - xx1, yy2 - yy1
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter + 1e-6
    iou = inter / union

    # --- Center distance penalty ---
    cx1, cy1 = (x1 + x2)/2, (y1 + y2)/2
    cx2, cy2 = (xx1 + xx2)/2, (yy1 + yy2)/2
    center_dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2

    # diagonal length of minimum enclosing box
    enc_x1, enc_y1 = min(x1, xx1), min(y1, yy1)
    enc_x2, enc_y2 = max(x2, xx2), max(y2, yy2)
    enc_diag_sq = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2 + 1e-6

    # --- Aspect ratio penalty ---
    v = (4 / (math.pi**2)) * (math.atan(w2/h2) - math.atan(w1/h1))**2

    if ALPHA_CIOU == None:
        alpha = v / (1 - iou + v + 1e-6)
    else:
        alpha = ALPHA_CIOU

    # --- CIoU score ---
    ciou_score = iou - (center_dist_sq / enc_diag_sq) - alpha * v
    return ciou_score

def box_to_mask(frame, cx, cy, w, h):
    H, W = frame.shape[:2]

    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)

    # clamp
    x1 = max(0, min(W-1, x1))
    x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1))
    y2 = max(0, min(H-1, y2))

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    return mask

def draw_box(frame, box, color, label):

    if box is None:
        return
    cx, cy, w, h = box
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1+5, y2+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    return

# -- Unused function -- #

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