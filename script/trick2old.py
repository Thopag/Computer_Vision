# ===========================================================
# TRICK2 — FINAL VERSION WITH
# - BALL COLOR CHANGE
# - BOTTLE CONTOUR GROW + SHRINK
# - MUSHROOM FADE OUT + FADE IN (FIRST 2 TOUCH BLOCKS ONLY)
# ===========================================================
import cv2
import numpy as np

from script.utils.detection.color import change_color_mask
from script.utils.mask_operation import roi
from utils.geometry import (
    grow_object,
    grow_object_magic,
    fade_object_magic,        # Already implemented in geometry.py
    find_contour_from_bbox    # Used inside grow/fade
)

# ===========================================================
# LOAD ready.txt
# ===========================================================
def load_ready(path):
    data = {}
    with open(path, "r") as f:
        next(f)
        for line in f:
            frame, tid, cx, cy, w, h = line.strip().split(",")
            frame = int(frame)
            tid   = int(tid)
            cx, cy, w, h = map(float, (cx, cy, w, h))

            if frame not in data:
                data[frame] = []

            data[frame].append({
                "id": tid,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h
            })
    return data


# ===========================================================
# LOAD interactions.txt
# ===========================================================
def load_interactions(path):

    ball_frames = []
    bottle_frames = []
    mushroom_frames = []

    with open(path, "r") as f:
        next(f)
        for line in f:
            if ":" not in line:
                continue
            frame, txt = line.strip().split(":")
            frame = int(frame)
            txt = txt.lower()

            if "ball" in txt:
                ball_frames.append(frame)
            elif "bottle" in txt:
                bottle_frames.append(frame)
            elif "mushroom" in txt:
                mushroom_frames.append(frame)

    return {
        "ball": sorted(ball_frames),
        "bottle": sorted(bottle_frames),
        "mushroom": sorted(mushroom_frames)
    }


# ===========================================================
# GROUP consecutive frames into blocks
# ===========================================================
def group_blocks(frames):
    if not frames:
        return []
    blocks = []
    curr = [frames[0]]
    for f in frames[1:]:
        if f == curr[-1] + 1:
            curr.append(f)
        else:
            blocks.append(curr)
            curr = [f]
    blocks.append(curr)
    return blocks


# ===========================================================
# SIMPLE BOX MASK
# ===========================================================
def box_to_mask(frame, cx, cy, w, h):
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


# ===========================================================
# MAIN TRICK2
# ===========================================================
def trick2(
    cap,
    writer,
    nb_frame,
    ready_path,
    interaction_path,
    file=None
):

    file.write("Start trick2\n")

    ready = load_ready(ready_path)
    inter = load_interactions(interaction_path)

    # Blocks
    ball_blocks     = group_blocks(inter["ball"])
    bottle_blocks   = group_blocks(inter["bottle"])
    mushroom_blocks = group_blocks(inter["mushroom"])

    ball_count     = 0
    bottle_count   = 0
    mushroom_count = 0

    def is_start_of_block(frame, blocks):
        for blk in blocks:
            if frame == blk[0]:
                return True
        return False

    # --------------------------------------------------------
    # BOTTLE intervals
    # --------------------------------------------------------
    if len(bottle_blocks) >= 1:
        bottle_grow_start = bottle_blocks[0][-1]
    else:
        bottle_grow_start = None

    if len(bottle_blocks) >= 2:
        bottle_grow_end = bottle_blocks[1][0]
    else:
        bottle_grow_end = None

    if len(bottle_blocks) >= 2:
        bottle_shrink_start = bottle_blocks[1][0]
        bottle_shrink_end   = bottle_shrink_start + 40
    else:
        bottle_shrink_start = None
        bottle_shrink_end   = None

    # --------------------------------------------------------
    # MUSHROOM fade intervals — ONLY first 2 blocks
    # --------------------------------------------------------
    if len(mushroom_blocks) >= 2:
        out_block = mushroom_blocks[0]
        in_block  = mushroom_blocks[1]

        fade_out_start = out_block[-1]
        fade_out_end   = in_block[0]

        fade_in_start  = in_block[0]
        fade_in_end    = in_block[-1]
    else:
        fade_out_start = fade_out_end = None
        fade_in_start  = fade_in_end = None

    # =======================================================
    # PROCESS FRAMES
    # =======================================================
    for frame_idx in range(nb_frame):

        ok, frame = cap.read()
        if not ok: break

        output = frame.copy()

        # Load objects
        wand = ball = bottle = mushroom = None
        if frame_idx in ready:
            for obj in ready[frame_idx]:
                if obj["id"] == 100: wand = obj
                elif obj["id"] == 2: ball = obj
                elif obj["id"] == 0: bottle = obj
                elif obj["id"] == 1: mushroom = obj

        # =======================================================
        # BALL LOGIC
        # =======================================================
        if ball_count < 3 and is_start_of_block(frame_idx, ball_blocks):
            ball_count += 1

        if ball is not None:
            ball_mask = box_to_mask(frame, ball["cx"], ball["cy"], ball["w"], ball["h"])
            if ball_count == 1:
                output = change_color_mask(output, [0,0,255], [255,0,0], ball_mask)
            elif ball_count == 2:
                output = change_color_mask(output, [0,0,255], [0,255,0], ball_mask)

        # =======================================================
        # BOTTLE GROW + SHRINK
        # =======================================================
        if bottle is not None:

            cx, cy, w, h = bottle["cx"], bottle["cy"], bottle["w"], bottle["h"]

            # Grow
            if bottle_grow_start and bottle_grow_end:
                if bottle_grow_start <= frame_idx <= bottle_grow_end:
                    output = grow_object_magic(
                        output, cx, cy, w, h,
                        frame_idx,
                        first_touch=bottle_grow_start,
                        last_touch=bottle_grow_end,
                        max_scale=3.0,
                        feather=9,
                        blur_amount=3
                    )
                    writer.write(output); continue

            # Shrink
            if bottle_shrink_start and bottle_shrink_end:
                if bottle_shrink_start <= frame_idx <= bottle_shrink_end:
                    output = grow_object_magic(
                        output, cx, cy, w, h,
                        frame_idx,
                        first_touch=bottle_shrink_end,
                        last_touch=bottle_shrink_start,
                        max_scale=3.0,
                        feather=19,
                        blur_amount=13
                    )
                    writer.write(output); continue

        # =======================================================
        # MUSHROOM FADE LOGIC
        # =======================================================
        if mushroom is not None:

            cx, cy, w, h = mushroom["cx"], mushroom["cy"], mushroom["w"], mushroom["h"]

            # Fade OUT
            if fade_out_start and fade_out_end:
                if fade_out_start <= frame_idx <= fade_out_end:
                    output = fade_object_magic(
                        output,
                        cx, cy, w, h,
                        frame_idx,
                        fade_start=fade_out_start,
                        fade_end=fade_out_end,
                        fade_in=False,
                        feather=17,
                        blur_amount=5
                    )

            # Fade IN
            if fade_in_start and fade_in_end:
                if fade_in_start <= frame_idx <= fade_in_end:
                    output = fade_object_magic(
                        output,
                        cx, cy, w, h,
                        frame_idx,
                        fade_start=fade_in_start,
                        fade_end=fade_in_end,
                        fade_in=True,
                        feather=17,
                        blur_amount=5
                    )

        writer.write(output)

    file.write("End trick2\n")
    return 1


# ===========================================================
# MAIN
# ===========================================================
def main():

    cap = cv2.VideoCapture("../input/dynamic/trick2.mp4")
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nb_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        "../output/trick2_result.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )

    f = open("../output/trick2_debug.txt", "w")
    f.write("Trick2 start\n")

    trick2(
        cap=cap,
        writer=writer,
        nb_frame=nb_frame,
        ready_path="../output/ready.txt",
        interaction_path="../output/interactions.txt",
        file=f
    )

    cap.release()
    writer.release()
    f.close()

    print("\n🎉 Trick2 finished successfully!")


# RUN
if __name__ == "__main__":
    main()
