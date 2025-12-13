import cv2
from script.CONFIG import FILE_NAME, IN_PATH
from script.utils.detection.color import detect_red_strict


def apply_red_mask_video(
    in_path,
    out_path,
    start_frame,
    end_frame = None
):
    cap = cv2.VideoCapture(in_path)

    if not cap.isOpened():
        raise IOError(f"Could not open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=False)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame == None:
        end_frame = total_frames

    frame_idx = 0

    for frame_idx in range(start_frame, end_frame+1):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 50 == 0:
            print(f"\rProcessing frame {frame_idx}/{total_frames}",end="")
        if frame_idx >= start_frame:
            mask = detect_red_strict(frame)
        else:
            # Before start_frame → fully black
            mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask[:] = 0

        out.write(mask)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved red mask video to: {out_path}")


if __name__ == "__main__":
    red_video_in  = IN_PATH
    red_video_out = f"files/wand_tracking/{FILE_NAME}_red_mask.mp4"

    apply_red_mask_video(
        in_path=red_video_in,
        out_path=red_video_out,
        start_frame=0 , # change this to tune when tracking starts
        end_frame= None 
    )
