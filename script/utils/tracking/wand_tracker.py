import cv2
import numpy as np

from script.CONFIG import *
from .kalman_filter import KalmanWandTracker
from .trajectory import object_trajectory
from script.utils.detection.wand_detector import wand_detector
from script.utils.box import draw_box


def wand_tracking(object_path, output_video, output_txt, start_frame, end_frame=None):

    cap = cv2.VideoCapture(IN_PATH)
    if not cap.isOpened():
        print(" ERROR: Cannot open input video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if end_frame == None:
        end_frame = total

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    log = open(output_txt, "w")
    log.write("frame,id,cx,cy,w,h\n")

    obj_trajs = object_trajectory(object_path)
    w_detector = wand_detector(obj_trajs)
    kalman_ready = False

    print("Tracking wand...")

    for frame_idx in range(start_frame, end_frame+1):

        ok, frame = cap.read()
        if not ok:
            break
        
        if kalman_ready:
            trk.decrement_timer()
            if trk.is_active():
                trk.predict()

        w_detector.detect(frame, frame_idx)
        if w_detector.have_something():

            if not kalman_ready:
                blobs_sorted = sorted(w_detector.blobs, key=lambda b:b["area"])
                chosen = blobs_sorted[len(blobs_sorted)//2]
                cx,cy = chosen["center"]
                trk = KalmanWandTracker(init_position=(cx, cy),time_before_sleep=KALMAN_WAND_TIMER)
                kalman_ready = True
            else:
                # want to minimize the distance
                lower_dist = np.inf
                for b in w_detector.blobs:
                    cx,cy = b["center"]
                    dist = np.hypot(cx-trk.px, cy-trk.py)
                    if dist < lower_dist:
                        lower_dist = dist
                        best_blob = b

                trk.update(best_blob)

        frame_with_pred = frame.copy()
        if kalman_ready:
            cx,cy,w,h = trk.kf.statePost[:4].ravel()

            w = max(w, MIN_W)
            h = max(h, MIN_H)

            if trk.is_active():
                color = (0,255,0)
                log.write(f"{frame_idx},-1,{cx},{cy},{w},{h}\n")
            else:
                color = (0,0,255)
            draw_box(frame_with_pred, (cx,cy,w,h), color, f"WAND")
    
        writer.write(frame_with_pred)
        print(f"Frame {frame_idx}/{end_frame}", end="\r")

    cap.release()
    writer.release()
    log.close()

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":

    interpolation_obj_txt = f"files/interpolation/{FILE_NAME}_{N_OBJECT}_obj.txt"
    wand_2_tracking_video = f"files/wand_tracking/{FILE_NAME}_trick2.mp4"
    wand_2_tracking_log =  f"files/wand_tracking/{FILE_NAME}_trick2.txt"


    wand_tracking(interpolation_obj_txt,wand_2_tracking_video,wand_2_tracking_log)
