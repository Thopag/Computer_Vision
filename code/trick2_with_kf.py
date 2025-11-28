import cv2
import numpy as np
import time
from utils.color import detect_color, change_color_mask
from utils.ROI import roi
from utils.geometry import grow_object


# ============================================================
# LOAD TRACKING LOG
# ============================================================
def load_file(txt_path):
    tracks = {}
    with open(txt_path, "r") as f:
        next(f)  # skip header

        for line in f:
            frame, tid, cx, cy, w, h = line.strip().split(",")
            frame = int(frame)
            tid = int(tid)
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


# Load tracking file
tracks = load_file("../output/trick2_log.txt")


# ============================================================
# UTILITY: Convert cx,cy,w,h → mask
# ============================================================
def mask_from_log(frame, cx, cy, w, h):
    H, W = frame.shape[:2]
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)

    x1 = max(0, min(W-1, x1))
    x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1))
    y2 = max(0, min(H-1, y2))

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


# ============================================================
# MAIN TRICK2 FUNCTION
# ============================================================
def trick2(cap, writer, nb_frame, file=None):

    file.write("Start trick2\n")
    start_time = time.time()

    # First tracked frame
    first_tracked_frame = min(tracks.keys())

    # Touch counters
    ball_touches = 0
    bottle_touches = 0
    mushroom_touches = 0

    # Rising edge memory
    prev_ball = False
    prev_bottle = False
    prev_mushroom = False

    # Reference masks for growing
    bottle_mask_ref = None
    mushroom_mask_ref = None

    TOUCH_THRESHOLD = 30

    for frame_idx in range(nb_frame):

        # -----------------------------------------------------------
        # Read next frame
        # -----------------------------------------------------------
        ok, frame = cap.read()
        if not ok:
            break

        output = frame.copy()

        # Progress display
        progress = frame_idx / nb_frame * 100
        print(f"\rProcessing: {progress:.2f}%", end="")

        # ===========================================================
        # SKIP FRAMES BEFORE TRACKING STARTS
        # ===========================================================
        if frame_idx < first_tracked_frame:
            writer.write(output)
            continue

        global_frame = frame_idx  # MATCHES tracking log numbers

        # ===========================================================
        # LOAD BOTTLE + MUSHROOM MASKS FROM LOG
        # ===========================================================
        bottle_mask = None
        mushroom_mask = None

        if global_frame in tracks:
            for obj in tracks[global_frame]:
                tid = obj["id"]
                cx, cy, w, h = obj["cx"], obj["cy"], obj["w"], obj["h"]

                if tid == 0:  # bottle
                    bottle_mask = mask_from_log(frame, cx, cy, w, h)
                    if bottle_mask_ref is None:
                        bottle_mask_ref = bottle_mask.copy()

                if tid == 1:  # mushroom
                    mushroom_mask = mask_from_log(frame, cx, cy, w, h)
                    if mushroom_mask_ref is None:
                        mushroom_mask_ref = mushroom_mask.copy()

        # ===========================================================
        # BUILD frame WITHOUT mushroom for wand detection
        # ===========================================================
        if mushroom_mask is not None:
            frame_no_mushroom = frame.copy()
            frame_no_mushroom[mushroom_mask > 0] = (0, 0, 0)
        else:
            frame_no_mushroom = frame

        # ===========================================================
        # WAND (RED TIP)
        # ===========================================================
        red_mask = detect_color(frame_no_mushroom,
                                [255, 0, 5],
                                [55, 55],
                                [255, 255],
                                tuning=25)
        red_mask = roi(red_mask, 3, 15)

        # ===========================================================
        # BALL (BLUE)
        # ===========================================================
        ball_mask = detect_color(frame,
                                 [0, 0, 255],
                                 [100, 100],
                                 [255, 255],
                                 tuning=25)
        ball_mask = roi(ball_mask, 10, 20)

        # ===========================================================
        # DRAW BOUNDING BOXES
        # ===========================================================
        def draw_mask_bbox(mask, color, label):
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                return
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        draw_mask_bbox(red_mask, (0, 0, 255), "WAND")
        draw_mask_bbox(ball_mask, (255, 0, 0), "BALL")
        if bottle_mask is not None:
            draw_mask_bbox(bottle_mask, (0, 255, 0), "BOTTLE")
        if mushroom_mask is not None:
            draw_mask_bbox(mushroom_mask, (255, 0, 255), "MUSHROOM")

        # ===========================================================
        # OVERLAP DETECTION (RISING EDGE)
        # ===========================================================

        # 1) BALL
        if ball_touches < 3:
            now = np.count_nonzero(cv2.bitwise_and(red_mask, ball_mask)) > TOUCH_THRESHOLD
            if now and not prev_ball:
                ball_touches += 1
            prev_ball = now

        # 2) BOTTLE
        elif bottle_touches < 2 and bottle_mask is not None:
            now = np.count_nonzero(cv2.bitwise_and(red_mask, bottle_mask)) > TOUCH_THRESHOLD
            if now and not prev_bottle:
                bottle_touches += 1
            prev_bottle = now

        # 3) MUSHROOM
        elif mushroom_touches < 2 and mushroom_mask is not None:
            now = np.count_nonzero(cv2.bitwise_and(red_mask, mushroom_mask)) > TOUCH_THRESHOLD
            if now and not prev_mushroom:
                mushroom_touches += 1
            prev_mushroom = now

        # ===========================================================
        # TRICK EFFECTS
        # ===========================================================
        # BALL COLOR CHANGE
        if ball_touches == 1:
            output = change_color_mask(output, [0,0,255], [255,0,0], ball_mask)
        elif ball_touches == 2:
            output = change_color_mask(output, [0,0,255], [0,255,0], ball_mask)

        # BOTTLE GROW
        if ball_touches >= 3 and bottle_touches < 2 and bottle_mask_ref is not None:
            output = grow_object(output, bottle_mask_ref, scale=3.0)

        # MUSHROOM GROW
        if ball_touches >= 3 and bottle_touches >= 2 and mushroom_touches < 2 and mushroom_mask_ref is not None:
            output = grow_object(output, mushroom_mask_ref, scale=3.0)

        # ===========================================================
        # TEXT OVERLAY
        # ===========================================================
        cv2.putText(output, f"Ball: {ball_touches}/3", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        cv2.putText(output, f"Bottle: {bottle_touches}/2", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        cv2.putText(output, f"Mushroom: {mushroom_touches}/2", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        writer.write(output)

    # END LOOP
    total = time.time() - start_time
    print(f"\nDone in {total:.2f} seconds.")

    file.write("End trick2\n")
    return 1



# ============================================================
# MAIN
# ============================================================
def main():

    cap = cv2.VideoCapture("../input/dynamic/trick2.mp4")
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nb_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter("../output/trick2_result.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))

    f = open("../output/trick2_debug.txt", "w")
    f.write("Trick2 start\n")

    trick2(cap, writer, nb_frame, file=f)

    cap.release()
    writer.release()
    f.close()

    print("🎉 Trick2 finished successfully!")


if __name__ == "__main__":
    main()
