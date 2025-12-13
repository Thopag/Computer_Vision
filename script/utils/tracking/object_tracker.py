import os
import cv2
import numpy as np
from .kalman_filter import KalmanBoxTracker
from ..box import xyxy_to_cxcywh, cxcywh_to_xyxy, ciou, draw_box
from ...CONFIG import *
from ..detection.yolo_detector import yolo_detector

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =====================================================================
# MOUSE CLICK SELECTION
# =====================================================================
clicks = []
current_frame = None

def mouse_callback(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x,y))
        print("Clicked:", (x,y))

def select_bboxes(frame, nbr_object):
    global clicks, current_frame
    clicks = []
    current_frame = frame.copy()

    cv2.namedWindow("Select Objects", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Objects", mouse_callback)

    print(f"\nSelect {nbr_object} objects: click TOP-LEFT then BOTTOM-RIGHT")

    while True:
        temp = current_frame.copy()
        for i in range(0, len(clicks), 2):
            if i+1 < len(clicks):
                cv2.rectangle(temp, clicks[i], clicks[i+1], (0,255,0), 2)
        cv2.imshow("Select Objects", temp)

        if len(clicks) == nbr_object*2:
            break
        if cv2.waitKey(20) == 27:
            break

    cv2.destroyWindow("Select Objects")

    boxes = []
    for i in range(0, len(clicks), 2):
        (x1,y1),(x2,y2) = clicks[i], clicks[i+1]
        boxes.append(xyxy_to_cxcywh((x1,y1,x2,y2)))
    return boxes


# =====================================================================
# MAIN
# =====================================================================
def object_tracking(out_path, log_path, nbr_object, start_frame, end_frame=None):

    try:
        cap = cv2.VideoCapture(IN_PATH)
        if not cap.isOpened():
            raise IOError("Cannot open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame == None:
            end_frame = total_frames

        # jump to selection frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame0 = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {start_frame}")

        # SELECT OBJECTS
        init_boxes = select_bboxes(frame0, nbr_object)

        # CREATE TRACKERS
        trackers = [KalmanBoxTracker(b, KALMAN_OBJ_TIMER) for b in init_boxes]

        # YOLO DETECTOR
        detector = yolo_detector(BLACKLIST, CONF_TRESHOLD, CONTROL_LIST)

        # OUTPUT VIDEO
        writer = cv2.VideoWriter(out_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (W,H))

        log = open(log_path, "w")
        log.write(f"nbr_object : {nbr_object}\n")
        log.write("Frame,ID,cx,cy,w,h\n")

        print("\n Online Kalman tracking...")

        # reset to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame+1):

            ok, frame = cap.read()
            if not ok:
                break

            detector.detect(frame)

            if detector.have_something():
                all_i_per_det = np.zeros( (len(detector.boxes_cc), nbr_object), dtype=float)
                for tid, trk in enumerate(trackers):
                    if trk.is_active():
                        trk.predict()

                    for deti, det in enumerate(detector.boxes_xy):
                        i = ciou(cxcywh_to_xyxy(trk.last_pred), det)
                        all_i_per_det[deti][tid] = i

                best_matches = []
                for i_vec, det in zip(all_i_per_det, detector.boxes_cc):
                    best_match = np.argmax(i_vec)
                    if best_match not in best_matches:
                        trackers[best_match].update(det)
                    best_matches.append(best_match)

            # DRAW
            frame_with_pred = frame.copy()
            for tid, trk in enumerate(trackers):

                cx,cy,w,h = trk.kf.statePost[:4].ravel()
                trk.decrement_timer()

                if trk.is_active():
                    color = (0,255,0)
                    log.write(f"{frame_idx},{tid},{cx},{cy},{w},{h}\n")
                else:
                    color = (0,0,255)
                draw_box(frame_with_pred, (cx,cy,w,h), color, f"ID {tid}")


            writer.write(frame_with_pred)
            print(f"Frame {frame_idx}/{end_frame}", end="\r")

        cap.release()
        writer.release()
        log.close()
        print("\nDONE. Tracking video saved:", out_path)

    except KeyboardInterrupt:
        print("\n CTRL+C pressed")

    finally:
        try: cap.release()
        except: pass
        try: writer.release()
        except: pass
        cv2.destroyAllWindows()
        print(" Clean exit")

def backup_tracking(out_path, log_path, nbr_object, start_frame, detect_each_x_frame, end_frame=None):

    try:
        cap = cv2.VideoCapture(IN_PATH)
        if not cap.isOpened():
            raise IOError("Cannot open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame == None:
            end_frame = total_frames

        # jump to selection frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame0 = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {start_frame}")

        # SELECT OBJECTS
        init_boxes = select_bboxes(frame0, nbr_object)

        # CREATE TRACKERS
        trackers = [KalmanBoxTracker(b, KALMAN_OBJ_TIMER) for b in init_boxes]

        # OUTPUT VIDEO
        writer = cv2.VideoWriter(out_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (W,H))

        log = open(log_path, "w")
        log.write(f"nbr_object : {nbr_object}\n")
        log.write("Frame,ID,cx,cy,w,h\n")

        print("\n Online Kalman tracking...")

        # reset to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame+1):

            ok, frame = cap.read()
            if not ok:
                break

            for trk in trackers:
                trk.decrement_timer()
                if trk.is_active():
                    trk.predict()

            if (frame_idx % detect_each_x_frame) == 0:
                manual_boxes = select_bboxes(frame, nbr_object)
    
                for box, trk in zip(manual_boxes, trackers):
                    trk.update(box)

            # DRAW
            frame_with_pred = frame.copy()
            for tid, trk in enumerate(trackers):
                cx,cy,w,h = trk.kf.statePost[:4].ravel()
                if trk.is_active():
                    color = (0,255,0)
                    log.write(f"{frame_idx},{tid},{cx},{cy},{w},{h}\n")
                else:
                    color = (0,0,255)
                draw_box(frame_with_pred, (cx,cy,w,h), color, f"ID {tid}")

            writer.write(frame_with_pred)
            print(f"Frame {frame_idx}/{end_frame}", end="\r")

        cap.release()
        writer.release()
        log.close()
        print("\nDONE. Tracking video saved:", out_path)

    except KeyboardInterrupt:
        print("\n CTRL+C pressed")

    finally:
        try: cap.release()
        except: pass
        try: writer.release()
        except: pass
        cv2.destroyAllWindows()
        print(" Clean exit")

def static_tracking(out_path, log_path, nbr_object, start_frame, end_frame=None):

    try:
        cap = cv2.VideoCapture(IN_PATH)
        if not cap.isOpened():
            raise IOError("Cannot open video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        log = open(log_path, "w")
        log.write(f"nbr_object : {nbr_object}\n")
        log.write("Frame,ID,cx,cy,w,h\n")

        if end_frame == None:
            end_frame = total_frames

        # jump to selection frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame0 = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {start_frame}")
    
        # SELECT OBJECTS
        init_boxes = select_bboxes(frame0, nbr_object)

        for i, box in enumerate(init_boxes):
            cx, cy, w, h = box
            log.write(f"{start_frame},{i},{cx},{cy},{w},{h}\n")

        # set to last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)

        ok, last_frame = cap.read()
        if not ok:
            raise IOError("Cannot open video")

        last_boxes = select_bboxes(last_frame, nbr_object)

        for i, box in enumerate(last_boxes):
            cx, cy, w, h = box
            log.write(f"{end_frame},{i},{cx},{cy},{w},{h}\n")

        cap.release()
        log.close()
        print("\nDONE. Tracking video saved:", out_path)

    except KeyboardInterrupt:
        print("\n CTRL+C pressed")

    finally:
        try: cap.release()
        except: pass
        cv2.destroyAllWindows()
        print(" Clean exit")

# =====================================================================
# RUN
# =====================================================================
if __name__ == "__main__":

    obj_tracking_video = f"files/object_tracking/{FILE_NAME}_{N_OBJECT}_obj.mp4"
    obj_tracking_log = f"files/object_tracking/{FILE_NAME}_{N_OBJECT}_obj.txt"

    object_tracking(obj_tracking_video, obj_tracking_log, 
                        N_OBJECT, OBJ_TRACKER_START_FRAME, end_frame=OBJ_TRACKER_END_FRAME)
    
    ball_tracking_video = f"files/object_tracking/{FILE_NAME}_ball.mp4"
    ball_tracking_log = f"files/object_tracking/{FILE_NAME}_ball.txt"

    #object_tracking(ball_tracking_video, ball_tracking_log, 
    #                1, OBJ_BALL_TRACKER_START_FRAME, end_frame=OBJ_BALL_TRACKER_END_FRAME)

    obj_backup_tracking_video = f"files/object_tracking/{FILE_NAME}_{N_OBJECT}_obj_backup.mp4"
    obj_backup_tracking_log = f"files/object_tracking/{FILE_NAME}_{N_OBJECT}_obj_backup.txt"
    detect_each_x_frame = 45

    #backup_tracking(obj_backup_tracking_video, obj_backup_tracking_log, N_OBJECT, 
    #                OBJ_TRACKER_START_FRAME, detect_each_x_frame, end_frame=OBJ_TRACKER_END_FRAME)
