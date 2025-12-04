import cv2
import os
import numpy as np
import math

from detection.yolo_detector import YOLODetector
from detection.magician_detector import MagicianDetector
from code.utils.video_splitting import load_json, time_to_seconds

from code.tracking.wand_tracker_florent import CONFIG, detect_red_strict, extract_red_blobs, create_kalman
from motion.wand_motion import WandMotion
from motion.ball_motion import BallMotion
from motion.magician_motion import MagicianMotion


def run_trick3(video_path, json_path, output_path, fixed=True):

    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

   
    # 0) Start time depuis le JSON
   
    start_str   = load_json(json_path, fixed, trick_id=3)
    start_frame = int(time_to_seconds(start_str) * fps)

   
    # 1) Détecteurs YOLO
   
    yolo = YOLODetector("yolov8s.pt")
    magician_detector = MagicianDetector()

   
    # Première frame pour initialiser la balle CLONÉE
   
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, first = cap.read()
    if not ok:
        print(" Cannot read first frame.")
        return

    detections = yolo.detect(first)

    # garder seulement sports ball + person
    filtered = [
        (cls, x1, y1, x2, y2, conf)
        for (cls, x1, y1, x2, y2, conf) in detections
        if cls.lower() in {"sports ball", "person"}
    ]

    # Balle initiale (référence)
    ball_box   = None
    best_conf  = 0.0

    for cls, x1, y1, x2, y2, conf in filtered:
        if cls.lower() == "sports ball" and conf > best_conf:
            best_conf = conf
            ball_box = (x1, y1, x2, y2)

    if ball_box is None:
        print(" Ball not found at start.")
        return

    bx1, by1, bx2, by2 = ball_box
    bw, bh   = bx2 - bx1, by2 - by1

    # position de référence (celle de YOLO au début)
    ball_ref_x = bx1 + bw // 2
    ball_ref_y = by1 + bh // 2

    # Patch et masque de la balle (clone)
    hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 80, 50])
    upper_blue = np.array([140, 255, 255])
    ball_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    roi_ball = first[by1:by1+bh, bx1:bx1+bw]
    roi_mask = ball_mask[by1:by1+bh, bx1:bx1+bw]
    mask_inv = cv2.bitwise_not(roi_mask)

   
    # 2) Redémarrer la vidéo à start_frame
   
    cap.release()
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

   
    # 3) Trackers / modele de mouvement
   
    kf = create_kalman(CONFIG)
    kalman_ready = False

    wand_motion      = WandMotion()
    ball_motion      = BallMotion(0.40, 0.30)   # juste pour sx/sy
    magician_motion  = MagicianMotion()

    # offsets cumulés du CLONE par rapport à la référence
    clone_dx = 0.0
    clone_dy = 0.0

    # pour lisser le mouvement de la wand
    prev_dx = 0.0
    prev_dy = 0.0

    MAGICIAN_STABLE_THRESHOLD = 10.0  # pixels/frame
    allow_ball_motion = False

    # position actuelle du clone (initialement = référence)
    clone_x = ball_ref_x
    clone_y = ball_ref_y

   
    # 4) Sortie vidéo
   
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    print("\n[INFO] Trick 3 running...\n")

    frame_id = start_frame

   
    # 5) BOUCLE PRINCIPALE
   
    while True:

        ok, frame = cap.read()
        if not ok:
            break

        print(f"\n========== FRAME {frame_id} ==========")

        
        # 5.1 YOLO : ball + magician (person) — DEBUG
        
        detections = yolo.detect(frame)
        filtered = [
            (cls, x1, y1, x2, y2, conf)
            for (cls, x1, y1, x2, y2, conf) in detections
            if cls.lower() in {"sports ball", "person"}
        ]

        # Ball réelle (pour debug, box verte)
        yolo_ball_box = None
        best_conf_ball = 0.0

        for cls, x1, y1, x2, y2, conf in filtered:
            if cls.lower() == "sports ball" and conf > best_conf_ball:
                best_conf_ball = conf
                yolo_ball_box = (x1, y1, x2, y2)

        if yolo_ball_box:
            x1, y1, x2, y2 = yolo_ball_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"BALL (YOLO) {best_conf_ball:.2f}",
                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            print(f"[BALL] YOLO box={yolo_ball_box} conf={best_conf_ball:.2f}")
        else:
            print("[BALL] not detected by YOLO on this frame")

        # Magicien
        magician_center, magician_box = magician_detector.detect(filtered)

        if magician_box:
            mx1, my1, mx2, my2 = magician_box
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 255, 255), 2)
            cv2.putText(frame, "MAGICIAN", (mx1, my1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        if magician_center:
            mdx, mdy = magician_motion.compute(magician_center)
            speed = math.hypot(mdx, mdy)
            print(f"[MAGICIAN] center={magician_center} speed={speed:.2f}")
            allow_ball_motion = speed < MAGICIAN_STABLE_THRESHOLD
        else:
            print("[MAGICIAN] not detected")
            allow_ball_motion = False

        print(f"[STATE] allow_ball_motion={allow_ball_motion}")

        
        # 5.2 Tracking de la wand (HSV + Kalman)
        
        mask_red = detect_red_strict(frame)
        blobs    = extract_red_blobs(mask_red, CONFIG)

        pred = kf.predict()
        px, py = int(pred[0, 0]), int(pred[1, 0])

        chosen = None
        if blobs:
            if not kalman_ready:
                chosen = max(blobs, key=lambda b: b["area"])
                cx, cy = chosen["center"]
                kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                kalman_ready = True
            else:
                best_d = 1e9
                for b in blobs:
                    cx, cy = b["center"]
                    d = math.hypot(cx - px, cy - py)
                    if d < best_d:
                        best_d = d
                        chosen = b

        if chosen:
            cx, cy = chosen["center"]
            xw, yw, ww, hw = chosen["bbox"]
            kf.correct(np.array([[cx], [cy]], dtype=np.float32))

            cv2.rectangle(frame, (xw, yw), (xw+ww, yw+hw), (0, 0, 255), 2)
            cv2.putText(frame, "WAND", (xw, yw-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            print(f"[WAND] DETECTED bbox={(xw,yw,ww,hw)} center={(cx,cy)}")
        else:
            cx, cy = px, py
            print(f"[WAND] LOST-> using prediction {(cx,cy)}")

        
        # 5.3 variation de la baguette -> lissé
        
        raw_dx, raw_dy = wand_motion.compute((cx, cy))

        # filtrage simple
        dx = 0.7 * prev_dx + 0.3 * raw_dx
        dy = 0.7 * prev_dy + 0.3 * raw_dy
        prev_dx, prev_dy = dx, dy

        print(f"[MOTION] wand raw=({raw_dx},{raw_dy}) smoothed=({dx:.2f},{dy:.2f})")

        
        # 5.4 Mouvement du clone de la balle
        amplify_x = 2.0 # Amplifie la distance parcourue par la clone ball
        amplify_y = 1.0
        
        old_clone_x = clone_x
        old_clone_y = clone_y

        if allow_ball_motion:
            # on accumule le mouvement en utilisant la sensibilite
            clone_dx += ball_motion.sx * dx * amplify_x 
            clone_dy += ball_motion.sy * dy * amplify_y

        # position absolue du clone
        clone_x = int(ball_ref_x + clone_dx)
        clone_y = int(ball_ref_y + clone_dy)

        print(f"[CLONE] offsets=({clone_dx:.1f},{clone_dy:.1f}) "
              f"pos from ({old_clone_x},{old_clone_y}) to ({clone_x},{clone_y})")

        
        # 5.5 Effacer l'ancien clone
        
        xo = old_clone_x - bw // 2
        yo = old_clone_y - bh // 2

        if 0 <= xo < width - bw and 0 <= yo < height - bh:
            roi_old = frame[yo:yo+bh, xo:xo+bw]
            roi_old_blur = cv2.GaussianBlur(roi_old, (7,7), 3)
            mean_color = cv2.mean(roi_old_blur, mask=mask_inv)[:3]
            frame[yo:yo+bh, xo:xo+bw] = np.full_like(roi_old, mean_color, dtype=np.uint8)

        
        # 5.6 Coller le clone à la new position
        
        xn = clone_x - bw // 2
        yn = clone_y - bh // 2

        if 0 <= xn < width - bw and 0 <= yn < height - bh:
            roi_dst = frame[yn:yn+bh, xn:xn+bw]
            bg = cv2.GaussianBlur(roi_dst, (7,7), 3)
            bg = cv2.bitwise_and(bg, bg, mask=mask_inv)
            fg = cv2.bitwise_and(roi_ball, roi_ball, mask=roi_mask)
            frame[yn:yn+bh, xn:xn+bw] = cv2.add(bg, fg)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    print("\n[INFO] Trick 3 complete.\n")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    run_trick3(
        video_path = os.path.join(base, "data", "video_group_4_fixed.mp4"),
        #video_path = os.path.join(base, "data", "group_11_fixed.mp4"),
        json_path  = os.path.join(base, "utils", "annotations_group_11.json"),
        output_path= os.path.join(base, "data", "test_trick3.mp4"),
        fixed      = True
    )
