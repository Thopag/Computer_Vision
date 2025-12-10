import cv2
import numpy as np

from script.utils.tracking.trajectory import object_trajectory, wand_trajectory
from script.utils.file_handler import load_interactions, group_blocks
from script.utils.box import box_to_mask, draw_box
from script.utils.color import change_color_mask
from script.utils.geometry import grow_object_magic
from script.CONFIG import *

# ============================================================================
#   BLUR EFFECT HELPERS (object ID = 2)
# ============================================================================
def find_contour_from_bbox(frame, cx, cy, w, h,
                           canny_low=15, canny_high=150,
                           blur_edges=3, dilate_iter=4, erode_iter=3,
                           min_area_ratio=0.001, max_area_ratio=0.95):

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
    gray = cv2.GaussianBlur(gray, (blur_edges, blur_edges), 0)

    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.dilate(edges, None, iterations=dilate_iter)
    edges = cv2.erode(edges, None, iterations=erode_iter)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None

    crop_area = crop.shape[0] * crop.shape[1]
    min_area = crop_area * min_area_ratio
    max_area = crop_area * max_area_ratio

    mask = np.zeros(edges.shape, dtype=np.uint8)
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            cv2.fillConvexPoly(mask, c, 255)

    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    mask = (mask > 20).astype(np.uint8) * 255
    return mask


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

    full_mask = np.zeros((H, W), dtype=np.uint8)
    if mask_crop is None:
        return full_mask

    full_mask[y1:y2, x1:x2] = mask_crop
    return full_mask


def selective_blur_with_mask(frame, full_mask):
    blurred = cv2.GaussianBlur(frame, (35, 35), 0)
    mask_3d = cv2.merge([full_mask, full_mask, full_mask]) / 255.0
    return (frame * mask_3d + blurred * (1 - mask_3d)).astype(np.uint8)


# ============================================================================
# EFFECT 1 — BALL COLOR (object ID = 0)
# ============================================================================
def apply_ball_color(frame, ball_box, count):
    cx, cy, w, h = ball_box
    mask = box_to_mask(frame, cx, cy, w, h)

    # 1st touch → gray
    if count == 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.merge([gray, gray, gray])
        return np.where(mask[..., None] == 255, gray, frame)

    # 2nd touch → green
    if count == 2:
        return change_color_mask(frame, [0,0,255], [0,255,0], mask)

    return frame


# ============================================================================
# EFFECT 2 — BOTTLE GROW/SHRINK (object ID = 1)
# ============================================================================
def apply_bottle_effect(frame, bottle_box, frame_idx,
                        grow_start, grow_end,
                        shrink_start, shrink_end,
                        bottle_touch_count):

    cx, cy, w, h = bottle_box

    # GROW only happens after 1st touch
    if grow_start is not None and grow_end is not None:
        if grow_start <= frame_idx <= grow_end:
            return grow_object_magic(
                frame, cx, cy, w, h,
                frame_idx,
                first_touch=grow_start,
                last_touch=grow_end,
                max_scale=3.0,
                feather=9,
                blur_amount=3
            )

    # SHRINK only happens after 2nd touch
    if bottle_touch_count >= 2:
        if shrink_start is not None and shrink_end is not None:
            if shrink_start <= frame_idx <= shrink_end:
                return grow_object_magic(
                    frame, cx, cy, w, h,
                    frame_idx,
                    first_touch=shrink_end,
                    last_touch=shrink_start,
                    max_scale=3.0,
                    feather=19,
                    blur_amount=13
                )

    return frame


# ============================================================================
# EFFECT 3 — THIRD OBJECT BLUR (object ID = 2)
# ============================================================================
def apply_third_blur(frame, cx, cy, w, h, frame_idx,
                     blur_in_start, blur_in_end,
                     blur_out_start, blur_out_end):

    mask_crop = find_contour_from_bbox(frame, cx, cy, w, h)
    full_mask = place_mask_on_frame(mask_crop, frame.shape, cx, cy, w, h)

    alpha = 0.0

    # BLUR-IN: start at FIRST FRAME of block1 → block2 start
    if blur_in_start is not None and blur_in_end is not None:
        if blur_in_start <= frame_idx <= blur_in_end:
            alpha = (frame_idx - blur_in_start) / (blur_in_end - blur_in_start)

    # BLUR-OUT: start at FIRST FRAME of block2 → block2 end
    if blur_out_start is not None and blur_out_end is not None:
        if blur_out_start <= frame_idx <= blur_out_end:
            alpha = 1 - (frame_idx - blur_out_start) / (blur_out_end - blur_out_start)

    alpha = np.clip(alpha, 0, 1)

    if alpha > 0:
        scaled_mask = (full_mask.astype(np.float32) * alpha).astype(np.uint8)
        return selective_blur_with_mask(frame, scaled_mask)

    return frame


# ============================================================================
# TRICK2 MAIN
# ============================================================================
def trick2(cap, writer, nb_frame,
           object_path, interaction_path, wand_path,
           file=None):

    if file:
        file.write("Start Trick2\n")

    traj = object_trajectory(object_path, 3)
    wand = wand_trajectory(wand_path)

    inter = load_interactions(interaction_path)

    ball_blocks   = group_blocks(inter[0])[:3]
    bottle_blocks = group_blocks(inter[1])[:2]
    third_blocks  = group_blocks(inter[2])[:2]

    print("\nBall blocks:", ball_blocks)
    print("Bottle blocks:", bottle_blocks)
    print("Third blocks:", third_blocks, "\n")

    # -------------------------------------------------------
    # Determine WHEN to draw bounding boxes
    # -------------------------------------------------------
    frames_objects = []
    for k in [0,1,2]:
        frames_objects += traj.obj_dict[k]["frame"]

    first_object_frame = min(frames_objects)
    first_wand_frame   = wand.wand_dict["frame"][0] if wand.wand_dict["frame"] else 10**9
    draw_start = min(first_object_frame, first_wand_frame)

    # -------------------------------------------------------
    # WHEN effects begin
    # -------------------------------------------------------
    all_inter_frames = inter[0] + inter[1] + inter[2]
    trick_start = min(all_inter_frames) if all_inter_frames else 10**9

    # -------------------------------------------------------
    # Bottle timing
    # -------------------------------------------------------
    
    if len(bottle_blocks) >= 1:
        first_touch_frame = bottle_blocks[0][0]
    else:
        first_touch_frame = None
    
    if len(bottle_blocks) >= 2:
        second_touch_frame = bottle_blocks[1][0]
    else:
        second_touch_frame = None
    
    # GROW runs first → until shrink begins
    bottle_grow_start = first_touch_frame
    bottle_grow_end   = second_touch_frame
    
    # SHRINK only after second touch
    if second_touch_frame is not None:
        bottle_shrink_start = second_touch_frame
        bottle_shrink_end   = second_touch_frame + 40
    else:
        bottle_shrink_start = bottle_shrink_end = None
    

    # -------------------------------------------------------
    # Third object blur timing
    # -------------------------------------------------------
    if len(third_blocks) == 2:
        blk1, blk2 = third_blocks
        blur_in_start  = blk1[0]      # FIRST frame of block1
        blur_in_end    = blk2[0]      # FIRST frame of block2
        blur_out_start = blk2[0]      # blur-out starts at block2 start
        blur_out_end   = blk2[-1]
    else:
        blur_in_start = blur_in_end = None
        blur_out_start = blur_out_end = None

    # Counters
    ball_touch_count = 0
    bottle_touch_count = 0
    third_touch_count = 0

    effect_messages = []

    def put_overlay(img):
        y = 30
        for msg in effect_messages[-6:]:
            cv2.putText(img, msg, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            y += 30

    # ============================================================================
    # MAIN LOOP
    # ============================================================================
    for frame_idx in range(nb_frame):
         
        ok, frame = cap.read()
        if not ok:
            break
        # LIVE PROGRESS PRINT
        if frame_idx % 30 == 0:
            print(f"\rProcessing frame {frame_idx}/{nb_frame}  ({nb_frame-frame_idx} left)", end="")
        output = frame.copy()

        boxes = traj.boxs_at_frame(frame_idx)
        ball_box, bottle_box, third_box = boxes

        wand_box = wand.box_at_frame(frame_idx)

        # Draw bounding boxes
        if frame_idx >= draw_start:

            if wand_box:
                draw_box(output, wand_box, (0,0,255), "WAND")

            if ball_box:
                draw_box(output, ball_box, (0,255,0), "BALL")

            if bottle_box:
                draw_box(output, bottle_box, (0,255,0), "BOTTLE")

            if third_box:
                draw_box(output, third_box, (0,255,0), "OBJECT3")

        # Apply effects
        if frame_idx >= trick_start:

            # BALL
            if frame_idx in inter[0]:
                if ball_touch_count < len(ball_blocks):
                    if frame_idx == ball_blocks[ball_touch_count][0]:
                        ball_touch_count += 1
                        effect_messages.append(f"Ball color change block {ball_touch_count}")

            if ball_box:
                output = apply_ball_color(output, ball_box, ball_touch_count)

            # BOTTLE
            if frame_idx in inter[1]:
                if bottle_touch_count < len(bottle_blocks):
                    if frame_idx == bottle_blocks[bottle_touch_count][0]:
                        bottle_touch_count += 1
                        effect_messages.append(f"Bottle touch {bottle_touch_count}")

            if bottle_box:
                output = apply_bottle_effect(
                    output, bottle_box, frame_idx,
                    bottle_grow_start, bottle_grow_end,
                    bottle_shrink_start, bottle_shrink_end,
                    bottle_touch_count
                )

            # THIRD OBJECT
            if frame_idx in inter[2]:
                if third_touch_count < len(third_blocks):
                    if frame_idx == third_blocks[third_touch_count][0]:
                        third_touch_count += 1
                        effect_messages.append(f"Object3 blur block {third_touch_count}")

            if third_box:
                cx, cy, w, h = third_box
                output = apply_third_blur(
                    output,
                    cx, cy, w, h,
                    frame_idx,
                    blur_in_start, blur_in_end,
                    blur_out_start, blur_out_end
                )

        cv2.putText(output,
                    f"Ball touches: {ball_touch_count}/3",
                    (20, output.shape[0] - 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)
        
        cv2.putText(output,
                    f"Bottle touches: {bottle_touch_count}/2",
                    (20, output.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)
        
        cv2.putText(output,
                    f"Object3 touches: {third_touch_count}/2",
                    (20, output.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)
        

        put_overlay(output)
        writer.write(output)

    if file:
        file.write("End Trick2\n")

    return 1



# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():

    video_path       = IN_PATH
    object_path      = f"files/interpolation/object_{FILE_NAME}.txt"
    interaction_path = f"files/object_&_wand/Interaction_object_&_wand_{FILE_NAME}.txt"
    wand_path        = f"files/interpolation/wand_{FILE_NAME}.txt"
    output_path      = f"files/trick2/trick2_result.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: cannot open video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )

    log_file = open("trick2_log.txt", "w")

    print("Running Trick2 on", N, "frames...")
    trick2(
        cap=cap,
        writer=writer,
        nb_frame=N,
        object_path=object_path,
        interaction_path=interaction_path,
        wand_path=wand_path,
        file=log_file
    )

    log_file.close()
    cap.release()
    writer.release()
    print("DONE! Saved to:", output_path)




if __name__ == "__main__":
    main()
