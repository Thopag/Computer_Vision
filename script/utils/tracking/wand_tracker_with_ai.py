import cv2
import numpy as np
import torch

from script.CONFIG import *
from script.utils.box import draw_box
from script.utils.tracking.kalman_filter import KalmanWandTracker

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_video_predictor import SAM2VideoPredictor


# ============================================================================
# CLICK SELECTION (same UX as object tracker)
# ============================================================================

clicks = []

def mouse_callback(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print("Clicked:", (x, y))

def select_click(frame):
    global clicks
    clicks = []

    cv2.namedWindow("Select Wand", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Wand", mouse_callback)

    print("\nClick ONCE on the wand")

    while len(clicks) == 0:
        cv2.imshow("Select Wand", frame)
        cv2.waitKey(50)

    cv2.destroyWindow("Select Wand")
    return clicks[0]


# ============================================================================
# MASK → BBOX
# ============================================================================

def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    w = int(x2 - x1)
    h = int(y2 - y1)

    return cx, cy, w, h


# ============================================================================
# MAIN
# ============================================================================

def wand_tracking_with_ai():

    # ---------------------------
    # Load SAM2 (CPU)
    # ---------------------------
    device = "cpu"

    sam = build_sam2(
        config_file="sam2_repo/sam2/configs/sam2_hiera_tiny.yaml",
        checkpoint="checkpoints/sam2_hiera_tiny.pt",
        device="cpu",
        apply_postprocessing=True
    )
    predictor = SAM2VideoPredictor(sam)

    # ---------------------------
    # Video
    # ---------------------------
    cap = cv2.VideoCapture(IN_PATH)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(input("Enter frame number where wand appears: "))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Cannot read start frame")

    # ---------------------------
    # CLICK INIT
    # ---------------------------
    click_pt = select_click(frame0)

    # ---------------------------
    # Init SAM2
    # ---------------------------
    predictor.reset()
    predictor.add_new_points(
        frame_idx=0,
        obj_id=1,
        points=np.array([[click_pt[0], click_pt[1]]]),
        labels=np.array([1])
    )

    # ---------------------------
    # Init Kalman
    # ---------------------------
    mask0 = predictor.predict(frame0)[1][1]
    bbox0 = mask_to_bbox(mask0)

    if bbox0 is None:
        raise RuntimeError("SAM2 failed to segment wand")

    cx, cy, w, h = bbox0

    trk = KalmanWandTracker(
        init_position=(cx, cy),
        time_before_sleep=KALMAN_WAND_TIMER
    )

    # ---------------------------
    # Output
    # ---------------------------
    writer = cv2.VideoWriter(
        f"files/wand_tracking/{FILE_NAME}_ai.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    log = open(f"files/wand_tracking/{FILE_NAME}_ai.txt", "w")
    log.write("frame,id,cx,cy,w,h\n")

    # ---------------------------
    # Loop
    # ---------------------------
    frame_idx = start_frame
    while True:

        ok, frame = cap.read()
        if not ok:
            break

        trk.decrement_timer()
        if trk.is_active():
            trk.predict()

        masks = predictor.predict(frame)[1]
        if 1 in masks:
            bbox = mask_to_bbox(masks[1])
            if bbox is not None:
                cx, cy, w, h = bbox
                trk.update({"center": (cx, cy)})

        # draw
        frame_out = frame.copy()
        cx, cy, w, h = trk.kf.statePost[:4].ravel()
        draw_box(frame_out, (cx, cy, w, h), (0, 255, 0), "WAND")

        writer.write(frame_out)
        log.write(f"{frame_idx},-1,{cx},{cy},{w},{h}\n")

        frame_idx += 1
        print(f"Frame {frame_idx}/{total}", end="\r")

    cap.release()
    writer.release()
    log.close()
    print("\nDONE")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    wand_tracking_with_ai()
