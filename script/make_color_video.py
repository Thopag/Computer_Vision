import cv2
from script.CONFIG import FILE_NAME, IN_PATH
from script.utils.detection.color import detect_red_strict, detect_green


def apply_color_mask_video(detect_color, out_path, start_frame, end_frame = None):
    cap = cv2.VideoCapture(IN_PATH)

    if not cap.isOpened():
        raise IOError(f"Could not open video: {IN_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=False)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame == None:
        end_frame = total_frames


    for frame_idx in range(start_frame, end_frame+1):
        ok, frame = cap.read()
        if not ok:
            break

        print(f"\rProcessing frame {frame_idx}/{end_frame-start_frame}",end="")

        if frame_idx >= start_frame:
            mask = detect_color(frame)
        else:
            # Before start_frame → fully black
            mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask[:] = 0

        writer.write(mask)

    cap.release()
    writer.release()
    print(f"Saved red mask video to: {out_path}")


if __name__ == "__main__":

    red_video_out = f"files/color_output/{FILE_NAME}_red_mask.mp4"

    apply_color_mask_video(detect_color=detect_red_strict, out_path=red_video_out, start_frame=0, end_frame= None)

    green_video_out = f"files/color_output/{FILE_NAME}_green_mask.mp4"

    #apply_color_mask_video(detect_color=detect_green, out_path=green_video_out, start_frame=0, end_frame= None)
