import cv2
import time
import numpy as np
from ultralytics import YOLO
from code.utils.tracking.boxTracker import *
from code.utils.tracking.wandTracker import *
from code.utils.preprocessing.loadFile import load_ready , load_wand_log
from code.utils.preprocessing.preprocessing import box_to_mask
from code.utils.ROI import roi
from code.CONFIG import CONFIG

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




# =====================================================================
#  Object Tracker
# =====================================================================
def boxTracker():

    try:
        start_frame = CONFIG["START_FRAME"]

        cap = cv2.VideoCapture(CONFIG["IN_PATH"])
        if not cap.isOpened():
            raise IOError("Cannot open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # jump to selection frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame0 = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {start_frame}")

        # SELECT OBJECTS
        init_boxes = select_bboxes(frame0)

        # CREATE TRACKERS
        trackers = [KalmanBoxTracker(b) for b in init_boxes]
        trackers_timers = np.array([CONFIG["nbr_frame_before_sleep"]] * CONFIG["N_OBJECTS"])
        trackers_last_pred = {}

        # YOLO MODEL
        model = YOLO(CONFIG["MODEL"])

        # OUTPUT VIDEO
        writer = cv2.VideoWriter(CONFIG["OUT_PATH"],
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (W,H))

        log = open(CONFIG["LOG_PATH"], "w")
        log.write("Frame,ID,cx,cy,w,h\n")

        print("\n Online Kalman tracking...")

        # reset to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # YOLO detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb, verbose=False)[0]
            labels = results.names
            detections = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                label = labels[class_id]
                if label not in CONFIG["blacklist"]:
                    if float(box.conf[0]) < CONFIG["CONF_THR"]:
                        continue
                    x1,y1,x2,y2 = box.xyxy[0]
                    detections.append(xyxy_to_cxcywh(float(x1),float(y1),float(x2),float(y2)))

            if len(detections) > 0:
                i_det = np.zeros( (len(detections), CONFIG["N_OBJECTS"]), dtype=float)
                for tid, trk in enumerate(trackers):
                    trackers_timers[tid] = trackers_timers[tid] - 1
                    if trackers_timers[tid] > 0:
                        pred = trk.predict()
                        trackers_last_pred[tid] = cxcywh_to_xyxy(pred)

                    for deti, det in enumerate(detections):
                        i = diou(trackers_last_pred[tid], cxcywh_to_xyxy(det))
                        i_det[deti][tid] = i

                best_matches = []
                for i_vec, det in zip(i_det, detections):
                    best_match = np.argmax(i_vec)
                    if best_match not in best_matches:
                        trackers[best_match].update(det)
                        trackers_timers[best_match] = CONFIG["nbr_frame_before_sleep"]
                    best_matches.append(best_match)

            # DRAW
            vis = frame.copy()
            for tid, trk in enumerate(trackers):
                
                if trackers_timers[tid] > 0:
                    cx,cy,w,h = trk.kf.statePost[:4].ravel()
                    log.write(f"{frame_idx},{tid},{cx},{cy},{w},{h}\n")

                    state = trk.kf.statePost[:4].ravel()
                    x1,y1,x2,y2 = cxcywh_to_xyxy(state)

                    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(vis,f"ID {tid}",(x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                else:
                    state = trk.kf.statePost[:4].ravel()
                    x1,y1,x2,y2 = cxcywh_to_xyxy(state)

                    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.putText(vis,f"ID {tid}",(x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)


            writer.write(vis)
            print(f"\rFrame {frame_idx}/{total}", end="")
            frame_idx += 1

        cap.release()
        writer.release()
        log.close()
        print("\nDONE. Tracking video saved:", CONFIG["OUT_PATH"])

    except KeyboardInterrupt:
        print("\n CTRL+C pressed")

    finally:
        try: cap.release()
        except: pass
        try: writer.release()
        except: pass
        cv2.destroyAllWindows()
        print(" Clean exit")



# ============================================================================
#  Wand TRACKER (YOUR ORIGINAL CODE + OBJECT REMOVAL)
# ============================================================================
def wandTracker() :
    removal_log_path = CONFIG["LOG_PATH"]
    output_video = CONFIG["OUT_PATH_Wand"]
    output_txt =  CONFIG["LOG_PATH_Wand"]
    start_frame=  CONFIG["START_FRAME"]
    # Load object-removal log
    obj_log = load_ready(removal_log_path)

    cap = cv2.VideoCapture(CONFIG["IN_PATH"])
    if not cap.isOpened():
        print(" ERROR: Cannot open input video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    log = open(output_txt, "w")
    log.write("frame,id,cx,cy,w,h\n")

    kf = create_kalman(CONFIG)
    kalman_ready = False
    last_pos = None

    results = []
    frame_id = 0

    print("Tracking wand...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_id < start_frame:
            writer.write(frame)
            frame_id += 1
            continue

        # ------------------------------------------------------
        # REMOVE ALL TRACKED OBJECTS BEFORE WAND DETECTION
        # ------------------------------------------------------
        objects_here = obj_log.get(frame_id, None)
        cleaned = remove_objects(frame, objects_here)

        # detect wand in cleaned frame
        mask = detect_red_strict(cleaned)
        blobs = extract_red_blobs(mask, CONFIG)

        # prediction
        pred = kf.predict()
        px, py = float(pred[0]), float(pred[1])

        if last_pos is not None:
            px = np.clip(px, last_pos[0]-CONFIG["max_prediction_jump"],
                              last_pos[0]+CONFIG["max_prediction_jump"])
            py = np.clip(py, last_pos[1]-CONFIG["max_prediction_jump"],
                              last_pos[1]+CONFIG["max_prediction_jump"])

        chosen = None

        if blobs:
            if not kalman_ready:
                blobs_sorted = sorted(blobs, key=lambda b:b["area"])
                chosen = blobs_sorted[len(blobs_sorted)//2]
                cx,cy = chosen["center"]
                kf.statePost = np.array([[cx],[cy],[0],[0]], dtype=np.float32)
                kalman_ready = True
            else:
                best = 999999
                for b in blobs:
                    cx,cy = b["center"]
                    d = np.hypot(cx-px, cy-py)
                    if d < best:
                        best = d
                        chosen = b

        if chosen is None:
            cx, cy = int(px), int(py)
            x = cx - CONFIG["bbox_size"]//2
            y = cy - CONFIG["bbox_size"]//2
            w = h = CONFIG["bbox_size"]
        else:
            cx,cy = chosen["center"]
            x,y,w,h = chosen["bbox"]
            kf.correct(np.array([[cx],[cy]], dtype=np.float32))

        last_pos = (cx,cy)

        # draw wand on ORIGINAL FRAME
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),2)
        cv2.putText(frame, "WAND", (x,y-8), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        writer.write(frame)

        log.write(f"{frame_id},100,{cx},{cy},{w},{h}\n")
        results.append({"frame":frame_id, "cx":cx, "cy":cy, "w":w, "h":h})

        frame_id += 1

    cap.release()
    writer.release()
    log.close()

    print(" Forward pass complete — running smoothing...")

    smoothed = backward_smooth(results, CONFIG)

    smoothed_txt = output_txt.replace(".txt", "_smoothed.txt")
    with open(smoothed_txt, "w") as f:
        f.write("frame,id,cx,cy,w,h\n")
        for s in smoothed:
            f.write(f"{s['frame']},100,{s['cx']},{s['cy']},{s['w']},{s['h']}\n")
    
    print("Generating smoothed video...")

    # --- Re-open video ---
    cap2 = cv2.VideoCapture(CONFIG["IN_PATH"])
    smoothed_video = output_video.replace(".mp4", "_smoothed.mp4")

    writer2 = cv2.VideoWriter(
        smoothed_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    # Prepare dict: frame → (cx,cy,w,h)
    smooth_dict = { t["frame"]: t for t in smoothed }

    fid = 0
    while True:
        ok, frame2 = cap2.read()
        if not ok:
            break

        if fid in smooth_dict:
            s = smooth_dict[fid]
            cx, cy, w, h = s["cx"], s["cy"], s["w"], s["h"]
            x = cx - w//2
            y = cy - h//2

            cv2.rectangle(frame2, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame2, "WAND (smoothed)", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        writer2.write(frame2)
        fid += 1

    cap2.release()
    writer2.release()

    print("Smoothed video saved at:", smoothed_video)

    print(" DONE!")
    print("Video:", output_video)
    print("Track:", output_txt)
    print("Smoothed:", smoothed_txt)

# ============================================================
# VISUALIZER
# ============================================================
def visualizer():
    
    video_path         = CONFIG["IN_PATH"]
    wand_path          = CONFIG["LOG_PATH_Wand"]
    object_path        = CONFIG["LOG_PATH"]
    output_video       = CONFIG["VIDEO_READY"]
    output_ready       = CONFIG["FINAL"]
    output_interactions= CONFIG["INTERACTION"]

    # Load logs
    wand_log = load_wand_log(wand_path)
    obj_log = load_ready(object_path)

    print(" Loaded logs.")
    print(f"  wand positions:   {len(wand_log)}")
    print(f"  object positions: {len(obj_log)}")

    # Open video
    cap = cv2.VideoCapture(video_path)
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

    ready = open(output_ready, "w")
    ready.write("frame,id,cx,cy,w,h\n")

    inter = open(output_interactions, "w")
    inter.write("interaction log\n")

    print("Processing…")

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

                elif obj["id"] == 1: # Object3
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
        draw_box(mushroom_box, (255, 0, 255), "OBJECT3")
        draw_box(ball_box,     (255, 0, 0),  "BALL")

        # -------------------------------------------------------------
        # INTERACTION DETECTION (expanded wand mask)
        # -------------------------------------------------------------
        if ball_box is not None and np.count_nonzero(ball_mask & wand_mask_expanded) > 20:
            inter.write(f"{frame_idx}: wand touches ball\n")

        if bottle_box is not None and np.count_nonzero(bottle_mask & wand_mask_expanded) > 20:
            inter.write(f"{frame_idx}: wand touches bottle\n")

        if mushroom_box is not None and np.count_nonzero(mushroom_mask & wand_mask_expanded) > 20:
            inter.write(f"{frame_idx}: wand touches Object3\n")

        # -------------------------------------------------------------
        # WRITE VISUAL FRAME
        # -------------------------------------------------------------
        writer.write(output)

    # ========================= END LOOP =========================

    cap.release()
    writer.release()
    ready.close()
    inter.close()

    print("\nVISUALIZER DONE")
    print("video :", output_video)
    print("ready :", output_ready)
    print("interactions :", output_interactions)




# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    start1 = time.time()
    boxTracker()
    end1 = time.time()
    wandTracker()
    end2 = time.time()
    visualizer()
    end3 =time.time()
    print("=============================================")
    elapsed_1 = end1 - start1
    print(f"boxTracker Execution time: {elapsed_1:.4f} seconds")
    elapsed_2 = end2 - end1
    print(f"wandTracker Execution time: {elapsed_2:.4f} seconds")
    elapsed_3 = end3 - end2
    print(f"visualize   Execution time: {elapsed_3:.4f} seconds")
    elapsed = end3 - start1
    print(f"The main execution time is :{elapsed:.4f}")
    print("=============================================")

  
