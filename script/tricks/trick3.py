import cv2
import numpy as np

from script.CONFIG import *
from script.utils.tracking.trajectory import object_trajectory, wand_trajectory


def trick3(cap, writer, nb_frame, frame_shift, object_path, wand_path, debug=False):

    # --------------------------------------------------
    # 1) Chargement des trajectoires OFFLINE (fichiers)
    # --------------------------------------------------

    traj = object_trajectory(object_path)
    wand = wand_trajectory(wand_path)

    # --------------------------------------------------
    # 2) Paramètres Trick3 (CONFIG)
    # --------------------------------------------------
    alpha = TRICK3_ALPHA

    amp_x = TRICK3_AMP_X
    amp_y_up   = TRICK3_AMP_Y_UP
    amp_y_down = TRICK3_AMP_Y_DOWN

    max_dx = TRICK3_MAX_DX
    max_dy = TRICK3_MAX_DY

    # Extrapolation wand absente
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_missing_frames = int(TRICK3_MAX_MISSING_SEC * fps)
    decay = TRICK3_MISSING_DECAY                       

    # --------------------------------------------------
    # 3) États internes (mémoire temporelle)
    # --------------------------------------------------
    prev_wand = None
    prev_dx = 0.0
    prev_dy = 0.0

    last_valid_dx = 0.0
    last_valid_dy = 0.0
    missing_count = 0

    clone_dx = 0.0
    clone_dy = 0.0

    # Debug trajectoire wand
    wand_history = []

    # --------------------------------------------------
    # 4) Boucle principale
    # --------------------------------------------------
    for frame_idx in range(frame_shift, nb_frame+frame_shift):

        ok, frame = cap.read()
        if not ok:
            break

        print(f"Trick 3 progress: {(frame_idx)/(nb_frame-frame_shift) *100:.2f} %", end="\r")

        h_img, w_img = frame.shape[:2]

        # Balle et wand proviennent des trajectoires OFFLINE
        ball_box = traj.boxs_at_frame(frame_idx)[0]
        wand_box = wand.box_at_frame(frame_idx)

        output = frame.copy()

        # --------------------------------------------------
        # 4.1) Mouvement wand : mesure si dispo, sinon extrapolation
        # --------------------------------------------------
        dx = 0.0
        dy = 0.0

        if wand_box is not None:
            cx, cy, _, _ = wand_box

            # Debug trajectoire : stocker point courant
            if debug:
                wand_history.append((int(cx), int(cy)))
                if len(wand_history) > 60:
                    wand_history.pop(0)

            if prev_wand is not None:
                raw_dx = cx - prev_wand[0]
                raw_dy = cy - prev_wand[1]

                # Anti-jump : limite les gros sauts instantanés
                raw_dx = np.clip(raw_dx, -max_dx, max_dx)
                raw_dy = np.clip(raw_dy, -max_dy, max_dy)

                # Lissage exponentiel (mouvement plus stable)
                dx = alpha * prev_dx + (1.0 - alpha) * raw_dx
                dy = alpha * prev_dy + (1.0 - alpha) * raw_dy

                prev_dx, prev_dy = dx, dy

                # Dernière vitesse valide (sert si wand disparaît)
                last_valid_dx = dx
                last_valid_dy = dy

            # Wand re-trouvée : on reset le compteur d'absence
            missing_count = 0
            prev_wand = (cx, cy)

        else:
            # Wand absente -> on extrapole le mouvement
            missing_count += 1

            if missing_count <= max_missing_frames:
                # On continue avec la dernière vitesse connue
                dx = last_valid_dx
                dy = last_valid_dy
            else:
                # Trop long : on amortit progressivement
                last_valid_dx *= decay
                last_valid_dy *= decay
                dx = last_valid_dx
                dy = last_valid_dy

        # --------------------------------------------------
        # 4.2) Clone + inpainting de la balle
        # --------------------------------------------------
        if ball_box is not None:
            bx, by, bw, bh = ball_box

            # Sécurité bornes image
            bx = max(0, bx)
            by = max(0, by)
            bw = min(bw, w_img - bx)
            bh = min(bh, h_img - by)

            bx, by, bw, bh = map(int, (bx, by, bw, bh))

            # Patch réel de la balle (clone)
            ball_patch = frame[by:by+bh, bx:bx+bw].copy()

            # Supprimer la balle originale (inpainting)
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), 255, -1)
            clean_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

            # Appliquer déplacement à la balle (cumulatif)
            clone_dx += amp_x * dx

            if dy < 0:
                clone_dy += amp_y_up * dy
            else:
                clone_dy += amp_y_down * dy

            bx_new = int(bx + clone_dx)
            by_new = int(by + clone_dy)

            # Clamp pour éviter sortir de l'image
            bx_new = max(0, min(w_img - bw, bx_new))
            by_new = max(0, min(h_img - bh, by_new))

            # Coller le clone sur le fond reconstruit
            output = clean_frame
            output[by_new:by_new+bh, bx_new:bx_new+bw] = ball_patch

        # --------------------------------------------------
        # 4.3) Debug : box wand + trajectoire + info état
        # --------------------------------------------------
        if debug:
            # Trajectoire wand
            for i in range(1, len(wand_history)):
                cv2.line(output, wand_history[i-1], wand_history[i], (255, 0, 0), 2)

            # Box wand
            if wand_box is not None:
                wx, wy, ww, wh = wand_box
                cv2.rectangle(output, (wx, wy), (wx + ww, wy + wh), (0, 0, 255), 2)

            # Text debug
            cv2.putText(
                output,
                f"dx={dx:.2f} dy={dy:.2f} missing={missing_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        writer.write(output)

    return


# --------------------------------------------------
# MAIN DEBUG (style Trick2)
# --------------------------------------------------
def main():

    video_path  = IN_PATH
    output_path = f"files/trick3/trick3_result.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    print("Running Trick3 on", N, "frames...")
    print("  video :", video_path)
    print("  output:", output_path)

    interpolation_ball_txt = f"files/interpolation/{FILE_NAME}_ball.txt"
    interpolation_wand_3_txt = f"files/interpolation/{FILE_NAME}_wand_trick3.txt"

    trick3(cap=cap, writer=writer, nb_frame=N, frame_shift=0, object_path=interpolation_ball_txt, wand_path=interpolation_wand_3_txt, debug=True)

    cap.release()
    writer.release()

    print("\nDONE! Saved to:", output_path)


if __name__ == "__main__":
    main()
