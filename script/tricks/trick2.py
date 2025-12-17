import cv2
import numpy as np

from script.utils.tracking.trajectory import object_trajectory, wand_trajectory
from script.utils.file_handler import load_interactions, group_blocks
from script.utils.box import box_to_mask, draw_box
from script.utils.detection.color import change_color_mask
from script.utils.geometry import grow_object_magic
from script.CONFIG import *


# ============================================================================
#  SKETCH OUTLINE EFFECT (Object ID = 2)
# ============================================================================
def sketch_overlay(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)

    edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_col = 255 - edges_col  # invert

    return cv2.addWeighted(frame, 1.0, edges_col, 0.7, 0)


# ============================================================================
# BLUR/CONTOUR HELPERS
# ============================================================================
def find_contour_from_bbox(frame, cx, cy, w, h):
    H, W = frame.shape[:2]
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)

    x1 = max(0, min(W-1, x1))
    x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1))
    y2 = max(0, min(H-1, y2))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    mask = cv2.dilate(edges, None, iterations=2)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    return (mask > 15).astype(np.uint8) * 255


def place_mask_on_frame(mask_crop, frame_shape, cx, cy, w, h):
    H, W = frame_shape[:2]
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)

    x1 = max(0, min(W-1, x1))
    x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1))
    y2 = max(0, min(H-1, y2))

    full = np.zeros((H, W), dtype=np.uint8)
    if mask_crop is not None:
        full[y1:y2, x1:x2] = mask_crop

    return full


# ============================================================================
# EFFECT FUNCTIONS
# ============================================================================
def apply_ball_color(frame, ball_box, count):
    cx, cy, w, h = ball_box
    mask = box_to_mask(frame, cx, cy, w, h)

    if count == 1:  # grayscale
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.merge([g, g, g])
        return np.where(mask[..., None] == 255, g, frame)

    if count == 2:  # green
        return change_color_mask(frame, [0,0,255], [0,255,0], mask)

    return frame


def apply_bottle_effect(frame, bottle_box, frame_idx,
                        grow_start, grow_end,
                        shrink_start, shrink_end,
                        bottle_touch_count):

    cx, cy, w, h = bottle_box

    # GROW
    if grow_start is not None:
        if grow_end is None or frame_idx <= grow_end:
            if frame_idx >= grow_start:
                return grow_object_magic(
                    frame, cx, cy, w, h, frame_idx,
                    first_touch=grow_start,
                    last_touch=(grow_end if grow_end else frame_idx),
                    max_scale=3.0, feather=9, blur_amount=3
                )

    # SHRINK after 2nd touch
    if bottle_touch_count >= 2:
        if shrink_start is not None and shrink_end is not None:
            if shrink_start <= frame_idx <= shrink_end:
                return grow_object_magic(
                    frame, cx, cy, w, h, frame_idx,
                    first_touch=shrink_end,
                    last_touch=shrink_start,
                    max_scale=3.0, feather=19, blur_amount=13
                )

    return frame


def apply_sketch_effect(frame, cx, cy, w, h, frame_idx,
                        in_start, in_end,
                        out_start, out_end):

    mask_crop = find_contour_from_bbox(frame, cx, cy, w, h)
    mask = place_mask_on_frame(mask_crop, frame.shape, cx, cy, w, h)

    sketch = sketch_overlay(frame)

    alpha = 0

    # fade-in
    if in_start is not None and in_end is not None:
        if in_start <= frame_idx <= in_end:
            alpha = (frame_idx - in_start) / (in_end - in_start)

    # fade-out
    if out_start is not None and out_end is not None:
        if out_start <= frame_idx <= out_end:
            alpha = 1 - (frame_idx - out_start) / (out_end - out_start)

    alpha = float(np.clip(alpha, 0, 1))
    if alpha <= 0:
        return frame

    mask_f = (mask.astype(np.float32) / 255)[..., None]
    return (frame * (1 - alpha * mask_f) + sketch * (alpha * mask_f)).astype(np.uint8)


# ============================================================================
# MAIN TRICK 2  
# ============================================================================
def trick2(cap, writer, nb_frame, frame_shift,
           object_path, interaction_path, wand_path,
           debug=True):

    """_summary_ :

    Returns:
        _type_: _description_
    """
    traj = object_trajectory(object_path)
    wand = wand_trajectory(wand_path)

    inter = load_interactions(interaction_path)

    # BLOCKS
    ball_blocks = group_blocks(inter[0])[:3]
    bottle_blocks = group_blocks(inter[1])
    third_blocks = group_blocks(inter[2])[:2]

    # Bottle filtering (keep first + last)
    if len(bottle_blocks) > 2:
        bottle_blocks = [bottle_blocks[0], bottle_blocks[-1]]
    else:
        bottle_blocks = bottle_blocks[:2]

    # draw_start
    object_frames = []
    for k in [0,1,2]:
        object_frames += traj.obj_dict[k]["frame"]

    first_obj = min(object_frames)
    first_wand = wand.wand_dict["frame"][0] if wand.wand_dict["frame"] else 10**9
    draw_start = min(first_obj, first_wand)

    # trick_start
    all_int = inter[0] + inter[1] + inter[2]
    trick_start = min(all_int) if all_int else 10**9

    # Bottle timing
    if len(bottle_blocks) == 2:
        b1, b2 = bottle_blocks
        bottle_grow_start = b1[0]
        bottle_grow_end   = b2[0]
        bottle_shrink_start = b2[0]
        bottle_shrink_end   = bottle_shrink_start + 40
    else:
        bottle_grow_start = bottle_grow_end = None
        bottle_shrink_start = bottle_shrink_end = None

    # Object 3 timing
    if len(third_blocks) == 2:
        t1, t2 = third_blocks
        s_in_start  = t1[0]
        s_in_end    = t2[0]
        s_out_start = t2[0]
        s_out_end   = t2[-1]
    else:
        s_in_start = s_in_end = s_out_start = s_out_end = None

    # Counters
    ball_touch = 0
    bottle_touch = 0
    third_touch = 0

    effect_messages = []

    def print_progress(f):
        if f % 50 == 0:
            print(f"Trick 2 progress: {(frame_idx)/(nb_frame-frame_shift) *100:.2f} %", end="\r")

    def put_overlay(img):
        if not debug:
            return
        y = 30
        for msg in effect_messages[-7:]:
            cv2.putText(img, msg, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,255), 2)
            y += 35

    # ============================================================================
    # MAIN LOOP
    # ============================================================================
    for frame_idx in range(frame_shift, nb_frame+frame_shift):

        print_progress(frame_idx)

        ok, frame = cap.read()
        if not ok:
            break

        output = frame.copy()

        ball_box, bottle_box, third_box = traj.boxs_at_frame(frame_idx)
        wand_box = wand.box_at_frame(frame_idx)

        # ---------------- DRAW BOXES ----------------
        if debug and frame_idx >= draw_start:
            if wand_box:   draw_box(output, wand_box, (0,0,255), "WAND")
            if ball_box:   draw_box(output, ball_box, (0,255,0), "BALL")
            if bottle_box: draw_box(output, bottle_box, (0,255,0), "BOTTLE")
            if third_box:  draw_box(output, third_box, (0,255,0), "OBJECT3")

        # ---------------- APPLY EFFECTS ----------------
        if frame_idx >= trick_start:

            # BALL
            if frame_idx in inter[0]:
                if ball_touch < len(ball_blocks):
                    if frame_idx == ball_blocks[ball_touch][0]:
                        ball_touch += 1
                        effect_messages.append(f"Ball touch #{ball_touch}")

            if ball_box:
                output = apply_ball_color(output, ball_box, ball_touch)

            # BOTTLE
            if frame_idx in inter[1]:
                if bottle_touch < len(bottle_blocks):
                    if frame_idx == bottle_blocks[bottle_touch][0]:
                        bottle_touch += 1
                        effect_messages.append(f"Bottle touch #{bottle_touch}")

            if bottle_box:
                output = apply_bottle_effect(
                    output, bottle_box, frame_idx,
                    bottle_grow_start, bottle_grow_end,
                    bottle_shrink_start, bottle_shrink_end,
                    bottle_touch
                )

            # OBJECT 3 — Sketch
            if frame_idx in inter[2]:
                if third_touch < len(third_blocks):
                    if frame_idx == third_blocks[third_touch][0]:
                        third_touch += 1
                        effect_messages.append(f"Object3 touch #{third_touch}")

            if third_box:
                cx, cy, w, h = third_box
                output = apply_sketch_effect(
                    output, cx, cy, w, h,
                    frame_idx,
                    s_in_start, s_in_end,
                    s_out_start, s_out_end
                )

        # ---------------- OVERLAY TEXT ----------------
        if debug:
            H = output.shape[0]
            cv2.putText(output, f"Ball touches: {ball_touch}/3",
                        (20, H-90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(output, f"Bottle touches: {bottle_touch}/2",
                        (20, H-60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(output, f"Object3 touches: {third_touch}/2",
                        (20, H-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            put_overlay(output)

        writer.write(output)

    return 1



# ============================================================================
# MAIN
# ============================================================================
def main():

    video_path       = IN_PATH
    object_path      = f"files/interpolation/object_{FILE_NAME}.txt"
    interaction_path = f"files/object_&_wand/Interaction_object_&_wand_{FILE_NAME}.txt"
    wand_path        = f"files/interpolation/wand_{FILE_NAME}.txt"
    output_path      = f"files/trick2/trick2_result.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))

    log = open("trick2_log.txt", "w")

    print("Running Trick2 on", N, "frames...")

    trick2(
        cap, writer, N,
        object_path, interaction_path, wand_path,
        file=log,
        debug=False         # CHANGE TO False for clean output
    )

    log.close()
    cap.release()
    writer.release()

    print("\nDONE! Saved to:", output_path)


if __name__ == "__main__":
    main()
