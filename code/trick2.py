# ===========================================================
# TRICK2 — FINAL VERSION WITH
# - BALL COLOR CHANGE
# - BOTTLE CONTOUR GROW + SHRINK
# - MUSHROOM FADE OUT + FADE IN (FIRST 2 TOUCH BLOCKS ONLY)
# ===========================================================
import cv2
import numpy as np
from utils.color import change_color_mask
from utils.loadFile import load_interactions, load_ready
from preprocessing import box_to_mask,group_blocks
from utils.geometry import (
    grow_object_magic,
    fade_object_magic,        # Already implemented in geometry.py
)

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
