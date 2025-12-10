import cv2
import numpy as np
from .trajectory import object_trajectory, wand_trajectory
from ...CONFIG import *
from ..box import draw_box

# ============================================================
# VISUALIZER
# ============================================================
def visualizer_object(video_path, object_path, output_video):

    # Load logs
    obj_trajs = object_trajectory(object_path, N_OBJECT)

    print("▶ Loaded logs.")

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

    print("▶ Processing…")

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

    print("\n🎉 VISUALIZER DONE")
    print("→ video :", output_video)

def visualizer_wand(video_path, wand_path, output_video):

    # Load logs
    wand_traj = wand_trajectory(wand_path)

    print("▶ Loaded logs.")

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

    print("▶ Processing…")

    for frame_idx in range(N):

        ok, frame = cap.read()
        if not ok:
            break

        print(f"{frame_idx}/{N}", end="\r")

        output = frame.copy()

        box = wand_traj.box_at_frame(frame_idx)
        draw_box(output, box, (0, 255, 0)  , "WAND")

        writer.write(output)

    # ========================= END LOOP =========================

    cap.release()
    writer.release()

    print("\n🎉 VISUALIZER DONE")
    print("→ video :", output_video)
# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    visualizer_wand(
        video_path=IN_PATH,
        wand_path=f"files/interpolation/wand_{FILE_NAME}.txt",
        output_video=f"files/interpolation/visual_wand_{FILE_NAME}.mp4",
    )
    print("🎉 Visualization complete!")
