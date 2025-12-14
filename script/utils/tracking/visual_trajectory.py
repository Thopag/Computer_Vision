import cv2
import numpy as np
from script.utils.tracking.trajectory import object_trajectory, wand_trajectory
from script.utils.box import draw_box,box_to_mask
from script.utils.mask_operation import roi
from script.CONFIG import *

def visualizer_object(object_path, output_video):
    """
    Make a video of the object trajectory .txt
    
    Args:
        object_path: Trajectory ?txt
        output_video: Video output
    """

    obj_trajs = object_trajectory(object_path)

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

    cap.release()
    writer.release()

    print("Video :", output_video)

def visualizer_wand(wand_path, output_video):
    """
    Make a video of the wand trajectory .txt
    
    Args:
        wand_path: Trajectory ?txt
        output_video: Video output
    """

    wand_traj = wand_trajectory(wand_path)

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

    for frame_idx in range(N):

        ok, frame = cap.read()
        if not ok:
            break

        print(f"{frame_idx}/{N}", end="\r")

        output = frame.copy()

        box = wand_traj.box_at_frame(frame_idx)
        draw_box(output, box, (0, 255, 0) , "WAND")

        writer.write(output)

    cap.release()
    writer.release()

    print("video :", output_video)


def visualizer_combined(
    wand_path,
    object_path,
    output_video,
    output_all_objects,
    output_interactions,
):
    """
    Visual both wand and object 
    and create the interaction files for trick 2
    
    Args:
        wand_path: Trajectory .txt of wand
        object_path: Trajectory .txt of the objects
        output_video: Video of the combine trajectories
        output_all_objects: Trajectory .txt of wand + object
        output_interactions: Interation .txt used for trick 2
    """

    wand_traj = wand_trajectory(wand_path)
    obj_trajs = object_trajectory(object_path) 

    if obj_trajs.nbr_object != 3:
        raise ValueError("Object file trajectory have not 3 objects")

    cap = cv2.VideoCapture(IN_PATH)
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

    for frame_idx in range(N):

        ok, frame = cap.read()
        if not ok:
            break

        print(f"{frame_idx}/{N}", end="\r")

        output = frame.copy()

        wand_box = wand_traj.box_at_frame(frame_idx)
        wand_mask_expanded = np.zeros((H, W), dtype=np.uint8)

        if wand_box is not None:
            cx, cy, w, h = wand_box

            # Write wand to the output file with id = -1
            allobj_file.write(f"{frame_idx},-1,{cx},{cy},{w},{h}\n")

            # Wand mask
            wand_mask = box_to_mask(frame, wand_box)

            # Expand wand mask using CONFIG constants
            wand_mask_expanded = roi(wand_mask, WAND_ITER, WAND_DILATE)

            # Draw wand rectangle in RED
            draw_box(output, wand_box, (0, 0, 255), "WAND")

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

        writer.write(output)

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

    interpolation_obj_txt = f"files/interpolation/{FILE_NAME}_{N_OBJECT}_obj.txt"
    interpolation_wand_2_txt = f"files/interpolation/{FILE_NAME}_wand_trick2.txt"

    interaction_txt = f"files/object_&_wand/{FILE_NAME}_interaction_object_&_wand.txt"
    interaction_video = f"files/object_&_wand/{FILE_NAME}_interaction_object_&_wand.mp4"
    output_all_objects = f"files/object_&_wand/{FILE_NAME}_all_object_&_wand.txt"

    visualizer_combined(interpolation_wand_2_txt, interpolation_obj_txt, 
                            interaction_video, output_all_objects, interaction_txt)
    
    interpolation_static_txt = f"files/interpolation/{FILE_NAME}_{N_OBJECT}_obj_static.txt"
    visual_obj_static = f"files/interpolation/{FILE_NAME}_visual_{N_OBJECT}_obj_static.mp4"

    #visualizer_object(interpolation_static_txt, visual_obj_static)

    interpolation_wand_2_txt = f"files/interpolation/{FILE_NAME}_wand_trick2.txt"
    interpolation_wand_2_video = f"files/interpolation/{FILE_NAME}_visual_wand_trick2.mp4"
    #visualizer_object(interpolation_wand_2_txt, interpolation_wand_2_video)
    
