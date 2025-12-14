import cv2
import numpy as np

from script.CONFIG import *
from script.utils.tracking.trajectory import object_trajectory, wand_trajectory

def trick3(cap, writer, nb_frame, frame_shift, object_path, wand_path, debug=False):

    """
    Load pre-process trajectories.
    Ball and wand trajectories are computed offline and interpolated,
    so one position is available for each frame
    """
    traj = object_trajectory(object_path)
    wand = wand_trajectory(wand_path)

    """
    Trick 3 parameters
    """
    alpha = TRICK3_ALPHA

    amp_x = TRICK3_AMP_X
    amp_y_up = TRICK3_AMP_Y_UP
    amp_y_down = TRICK3_AMP_Y_DOWN

    max_dx = TRICK3_MAX_DX
    max_dy = TRICK3_MAX_DY

    max_dy_jump = TRICK3_MAX_DY_JUMP
    bad_dy_scale = TRICK3_BAD_DY_SCALE

    suspect_w = TRICK3_WAND_SUSPECT_W
    suspect_h = TRICK3_WAND_SUSPECT_H
    bad_box_scale = TRICK3_BAD_BOX_SCALE

    mask_k = TRICK3_MASK_KERNEL
    inpaint_r = TRICK3_INPAINT_RADIUS

    """
    Internal state kept between frames
    """
    prev_wand = None

    prev_dx = 0.0
    prev_dy = 0.0

    prev_dy_filtered = 0.0

    clone_dx = 0.0
    clone_dy = 0.0

    wand_history = []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_k, mask_k))

    """
    Main loop over frames
    """
    for frame_idx in range(frame_shift, frame_shift + nb_frame):

        ok, frame = cap.read()
        if not ok:
            break

        print(f"Trick 3 progress: {(frame_idx-frame_shift)/(nb_frame) *100:.2f} %", end="\r")

        h_img, w_img = frame.shape[:2]
        output = frame.copy()

        """
        Get bounding boxes from offline trajectories
        """
        ball_box = traj.boxs_at_frame(frame_idx)[0]
        wand_box = wand.box_at_frame(frame_idx)

        """
        Compute wand motion
        """
        dx = 0.0
        dy = 0.0
        bad_frame = False

        if wand_box is not None:
            cx, cy, ww, wh = wand_box

            if debug:
                wand_history.append((int(cx), int(cy)))
                if len(wand_history) > 60:
                    wand_history.pop(0)

            if prev_wand is not None:
                """
                Raw displacement of the wand
                """
                raw_dx = cx - prev_wand[0]
                raw_dy = cy - prev_wand[1]

                """
                Limit very large wand movements between frames
                """
                raw_dx = float(np.clip(raw_dx, -max_dx, max_dx))
                raw_dy = float(np.clip(raw_dy, -max_dy, max_dy))

                """
                Exponential moving average to smooth motion
                """
                dx = alpha * prev_dx + (1.0 - alpha) * raw_dx
                dy = alpha * prev_dy + (1.0 - alpha) * raw_dy

                prev_dx = dx
                prev_dy = dy

                """
                Detect abnormal vertical motion.
                """
                if abs(dy - prev_dy_filtered) > max_dy_jump:
                    bad_frame = True
                    dy *= bad_dy_scale

                """
                Large bounding box usually indicates a tracking error
                """
                if ww > suspect_w or wh > suspect_h:
                    bad_frame = True
                    dx *= bad_box_scale
                    dy *= bad_box_scale

                prev_dy_filtered = dy

            prev_wand = (cx, cy)

        """
        Remove the real ball and move the cloned one
        """
        if ball_box is not None:
            cx, cy, bw, bh = ball_box

            bw = int(bw)
            bh = int(bh)

            bx = int(cx - bw / 2)
            by = int(cy - bh / 2)

            """
            Limite coordinate to stay inside the image
            """
            bx = max(0, min(w_img - bw, bx))
            by = max(0, min(h_img - bh, by))

            ball_patch = frame[by:by + bh, bx:bx + bw].copy()
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), 255, -1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            output = cv2.inpaint(output, mask, inpaint_r, cv2.INPAINT_TELEA)

            """
            Accumulate displacement for the telekinesis effecf
            """
            clone_dx += amp_x * dx
            clone_dy += (amp_y_up if dy < 0 else amp_y_down) * dy

            bx_new = int(bx + clone_dx)
            by_new = int(by + clone_dy)

            bx_new = max(0, min(w_img - bw, bx_new))
            by_new = max(0, min(h_img - bh, by_new))

            output[by_new:by_new + bh, bx_new:bx_new + bw] = ball_patch

        """
        Debug visualization
        """
        if debug:
            for i in range(1, len(wand_history)):
                cv2.line(
                    output,
                    wand_history[i - 1],
                    wand_history[i],
                    (255, 0, 0),
                    2
                )

            cv2.putText(
                output,
                f"dx={dx:.2f} dy={dy:.2f} bad={int(bad_frame)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        writer.write(output)

    return

def main():

    video_path = IN_PATH
    output_path = f"files/trick3/trick3_result.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: cannot open video")
        return
    
    print("Start trick 3 only:")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    interpolation_ball_txt = f"files/interpolation/{FILE_NAME}_ball.txt"
    interpolation_wand_3_txt = f"files/interpolation/{FILE_NAME}_wand_trick3.txt"

    trick3(
        cap=cap,
        writer=writer,
        nb_frame=N,
        frame_shift=0,
        object_path=interpolation_ball_txt,
        wand_path=interpolation_wand_3_txt,
        debug=True
    )

    cap.release()
    writer.release()

    print("\nDONE! Saved to:", output_path)


if __name__ == "__main__":
    main()
