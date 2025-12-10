import cv2
import numpy as np
from script.utils.tracking.trajectory import object_trajectory, wand_trajectory
from script.utils.box import draw_box,box_to_mask
from script.utils.mask_operation import roi
from script.CONFIG import *
# ============================================================
# VISUALIZER
# ============================================================
def visualizer_object(object_path, output_video):

    # Load logs
    obj_trajs = object_trajectory(object_path, N_OBJECT)

    print(" Loaded logs.")

    # Open video
    cap = cv2.VideoCapture(IN_PATH)
    if not cap.isOpened():
        print(" ERROR: Cannot open input video")
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

    print(" Processing…")

    for frame_idx in range(N):

        ok, frame = cap.read()
        if not ok:
            break

        print(f"{frame_idx}/{N}", end="\r")

        output = frame.copy()

        boxes = obj_trajs.boxs_at_frame(frame_idx)

        for i, box in enumerate(boxes):
            draw_box(output, box, (0, 255, 0)  , str(i))

        writer.write(output)

    # ========================= END LOOP =========================

    cap.release()
    writer.release()

    print("\n VISUALIZER DONE")
    print(" video :", output_video)

def visualizer_wand(wand_path, output_video):

    # Load logs
    wand_traj = wand_trajectory(wand_path)

    print(" Loaded logs.")

    # Open video
    cap = cv2.VideoCapture(IN_PATH)
    if not cap.isOpened():
        print(" ERROR: Cannot open input video")
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

    print(" Processing…")

    for frame_idx in range(N):

        ok, frame = cap.read()
        if not ok:
            break

        print(f"{frame_idx}/{N}", end="\r")

        output = frame.copy()

        box = wand_traj.box_at_frame(frame_idx)
        draw_box(output, box, (0, 255, 0) , "WAND")

        writer.write(output)

    # ========================= END LOOP =========================

    cap.release()
    writer.release()

    print("\n VISUALIZER DONE")
    print("→ video :", output_video)

# ============================================================
# VISUALIZER Both + Generate interaction files
# ============================================================
def visualizer_combined(
    video_path,
    wand_path,
    object_path,
    output_video,
    output_all_objects,
    output_interactions,
):

    # ---------------------------------------------------------------
    # LOAD TRAJECTORIES
    # ---------------------------------------------------------------
    wand_traj = wand_trajectory(wand_path)
    obj_trajs = object_trajectory(object_path, 3)   # exactly 3 objects

    print(" Loaded logs.")
    print(f"  wand entries:   {len(wand_traj.wand_dict['frame'])}")
    print(f"  objects tracked: 3")

    # ---------------------------------------------------------------
    # OPEN VIDEO
    # ---------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" ERROR: cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )

    allobj_file = open(output_all_objects, "w")
    allobj_file.write("frame,id,cx,cy,w,h\n")

    inter_file = open(output_interactions, "w")
    inter_file.write("interaction log\n")

    print(" Processing…")

    # ---------------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------------
    for frame_idx in range(N):

        ok, frame = cap.read()
        if not ok:
            break

        output = frame.copy()

        # ===========================================================
        # WAND DATA + EXPANDED MASK
        # ===========================================================
        wand_box = wand_traj.box_at_frame(frame_idx)
        wand_mask_expanded = np.zeros((H, W), dtype=np.uint8)

        if wand_box is not None:
            cx, cy, w, h = wand_box

            # Write wand to the output file with id = -1
            allobj_file.write(f"{frame_idx},-1,{cx},{cy},{w},{h}\n")

            # Wand mask
            wand_mask = box_to_mask(frame, cx, cy, w, h)

            # Expand wand mask using CONFIG constants
            wand_mask_expanded = roi(wand_mask, WAND_ITER, WAND_DILATE)

            # Draw wand rectangle in RED
            draw_box(output, wand_box, (0, 0, 255), "WAND")

        # ===========================================================
        # OBJECT DATA + EXPANDED MASKS
        # ===========================================================
        obj_boxes = obj_trajs.boxs_at_frame(frame_idx)
        obj_masks = obj_trajs.masks_at_frame(frame_idx, frame)

        expanded_obj_masks = []

        for obj_id, (box, mask) in enumerate(zip(obj_boxes, obj_masks)):

            if box is None:
                expanded_obj_masks.append(None)
                continue

            cx, cy, w, h = box

            # Write object row to file
            allobj_file.write(f"{frame_idx},{obj_id},{cx},{cy},{w},{h}\n")

            # Dilate object mask for robustness
            expanded_mask = roi(mask, OBJ_ITER, OBJ_DILATE)
            expanded_obj_masks.append(expanded_mask)

            # Draw object rectangle in GREEN
            draw_box(output, box, (0, 255, 0), f"OBJ {obj_id}")

        # ===========================================================
        # INTERACTION DETECTION
        # ===========================================================
        if wand_box is not None:
            for obj_id, mask in enumerate(expanded_obj_masks):
                if mask is None:
                    continue

                overlap = np.count_nonzero(mask & wand_mask_expanded)

                # Same threshold as before, or adjust if needed
                if overlap > 20:
                    inter_file.write(
                        f"{frame_idx}   wand touches object {obj_id}\n"
                    )

        # ===========================================================
        # WRITE FRAME
        # ===========================================================
        writer.write(output)

    # ---------------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------------
    cap.release()
    writer.release()
    allobj_file.close()
    inter_file.close()

    print("\n DONE")
    print(" video:", output_video)
    print(" all objects:", output_all_objects)
    print(" interactions:", output_interactions)

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    # visualizer_object(
    #     video_path= IN_PATH,
    #     object_path= f"files/interpolation/object_{FILE_NAME}.txt",
    #     output_video= f"files/interpolation/visual_object_{FILE_NAME}.mp4")
    # #visualizer_wand(
    #     video_path=IN_PATH,
    #     wand_path=f"files/interpolation/wand_{FILE_NAME}.txt",
    #     output_video=f"files/interpolation/visual_wand_{FILE_NAME}.mp4")
    
    visualizer_combined(
    video_path = IN_PATH,
    wand_path = f"files/interpolation/wand_{FILE_NAME}.txt",
    object_path = f"files/interpolation/object_{FILE_NAME}.txt",
    output_video = f"files/object_&_wand/visual_object_&_wand_{FILE_NAME}.mp4",
    output_all_objects  = f"files/object_&_wand/visual_object_&_wand_{FILE_NAME}.txt",
    output_interactions = f"files/object_&_wand/Interaction_object_&_wand_{FILE_NAME}.txt")
    print(" Visualization complete!")
