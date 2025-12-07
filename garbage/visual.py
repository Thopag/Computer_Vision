import cv2
import numpy as np
import time
from utils.ROI import roi   # ★ added


# ============================================================
# LOAD OBJECT LOG (bottle + mushroom + ball)
# ============================================================
def load_object_log(path):
    tracks = {}
    with open(path, "r") as f:
        next(f)
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


# ============================================================
# LOAD WAND LOG (smoothed KF)
# ============================================================
def load_wand_log(path):
    wand = {}
    with open(path, "r") as f:
        next(f)
        for line in f:
            frame, tid, cx, cy, w, h = line.strip().split(",")
            frame = int(frame)
            cx, cy, w, h = map(float, (cx, cy, w, h))
            wand[frame] = {"cx": cx, "cy": cy, "w": w, "h": h}
    return wand


# ============================================================
# Converting box (cx,cy,w,h) → mask
# ============================================================
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


# ============================================================
# VISUALIZER
# ============================================================
def visualizer(
    video_path,
    wand_path,
    object_path,
    output_video,
    output_ready,
    output_interactions
):

    # Load logs
    wand_log = load_wand_log(wand_path)
    obj_log = load_object_log(object_path)

    print("▶ Loaded logs.")
    print(f"  wand positions:   {len(wand_log)}")
    print(f"  object positions: {len(obj_log)}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ ERROR: Cannot open input video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )

    ready = open(output_ready, "w")
    ready.write("frame,id,cx,cy,w,h\n")

    inter = open(output_interactions, "w")
    inter.write("interaction log\n")

    print("▶ Processing…")

    # ========================= LOOP FRAMES =========================
    for frame_idx in range(N):

        ok, frame = cap.read()
        if not ok:
            break

        output = frame.copy()

        # -------------------------------------------------------------
        # WAND MASK (expanded ROI for interaction)
        # -------------------------------------------------------------
        wand_mask_expanded = np.zeros((H, W), dtype=np.uint8)
        wand_box = None

        if frame_idx in wand_log:
            w = wand_log[frame_idx]
            wand_box = (w["cx"], w["cy"], w["w"], w["h"])

            # small bounding box
            base_mask = box_to_mask(frame, w["cx"], w["cy"], w["w"], w["h"])

            # ★ EXPAND wand region for robust overlap detection
            wand_mask_expanded = roi(base_mask, 3, 20)

            # write ORIGINAL wand box to ready.txt
            ready.write(f"{frame_idx},100,{w['cx']},{w['cy']},{w['w']},{w['h']}\n")

        # -------------------------------------------------------------
        # OBJECT MASKS (bottle=0, mushroom=1, ball=2)
        # -------------------------------------------------------------
        bottle_box = mushroom_box = ball_box = None
        bottle_mask = mushroom_mask = ball_mask = np.zeros((H, W), dtype=np.uint8)

        if frame_idx in obj_log:
            for obj in obj_log[frame_idx]:

                cx, cy, w, h = obj["cx"], obj["cy"], obj["w"], obj["h"]

                if obj["id"] == 0:   # bottle
                    bottle_box = (cx, cy, w, h)
                    bottle_mask = box_to_mask(frame, cx, cy, w, h)
                    ready.write(f"{frame_idx},0,{cx},{cy},{w},{h}\n")

                elif obj["id"] == 1: # mushroom
                    mushroom_box = (cx, cy, w, h)
                    mushroom_mask = box_to_mask(frame, cx, cy, w, h)
                    ready.write(f"{frame_idx},1,{cx},{cy},{w},{h}\n")

                elif obj["id"] == 2: # ball
                    ball_box = (cx, cy, w, h)
                    ball_mask = box_to_mask(frame, cx, cy, w, h)
                    ready.write(f"{frame_idx},2,{cx},{cy},{w},{h}\n")

        # -------------------------------------------------------------
        # DRAW BOXES
        # -------------------------------------------------------------
        def draw_box(box, color, label):
            if box is None:
                return
            cx, cy, w, h = box
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        draw_box(wand_box,     (0, 0, 255), "WAND")
        draw_box(bottle_box,   (0, 255, 0), "BOTTLE")
        draw_box(mushroom_box, (255, 0, 255), "MUSHROOM")
        draw_box(ball_box,     (255, 0, 0),  "BALL")

        # -------------------------------------------------------------
        # INTERACTION DETECTION (expanded wand mask)
        # -------------------------------------------------------------
        if ball_box is not None and np.count_nonzero(ball_mask & wand_mask_expanded) > 20:
            inter.write(f"{frame_idx}: wand touches ball\n")

        if bottle_box is not None and np.count_nonzero(bottle_mask & wand_mask_expanded) > 20:
            inter.write(f"{frame_idx}: wand touches bottle\n")

        if mushroom_box is not None and np.count_nonzero(mushroom_mask & wand_mask_expanded) > 20:
            inter.write(f"{frame_idx}: wand touches mushroom\n")

        # -------------------------------------------------------------
        # WRITE VISUAL FRAME
        # -------------------------------------------------------------
        writer.write(output)

    # ========================= END LOOP =========================

    cap.release()
    writer.release()
    ready.close()
    inter.close()

    print("\n🎉 VISUALIZER DONE")
    print("→ video :", output_video)
    print("→ ready :", output_ready)
    print("→ interactions :", output_interactions)



# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    visualizer(
        video_path="../input/dynamic/trick2.mp4",
        wand_path="../output/wand_tracker_final_smoothed.txt",
        object_path="../output/trick2_log.txt",
        output_video="../output/trick2_visual.mp4",
        output_ready="../output/ready.txt",
        output_interactions="../output/interactions.txt"
    )
    print("🎉 Visualization complete!")
