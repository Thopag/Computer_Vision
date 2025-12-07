import cv2
import os
import numpy as np

def detect_color(image, color ,lower,upper, tuning=25):
    """
    Detect a specific color in an image and create a mask.

    Args:
        image (np.array) : and input image in BGR format
        color (tuple): a list RGB color like [255,0,0]
        lower,upper (tuple) :  lower and upper value of s & v in hsv format
        tuning (int, optional):tuning parameter. Defaults to 25.

    Returns:
        (np.array): mask corresponding to the detected color
        
    """
    lower_s = lower[0]
    lower_v = lower[1]
    upper_s = upper[0]
    upper_v = upper[1]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color = np.uint8([[color]])
    
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_color)
    H = h[0][0]
    lower_h = max(H - tuning, 0)
    upper_h = min(H + tuning, 179)
    lower = np.array([lower_h, lower_s, lower_v])
    upper = np.array([upper_h, upper_s, upper_v])
    
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def change_color_mask(image, orig_rgb, target_rgb, mask):
    """
    Change a specific color in an image to a target color based on the mask given as input.

    Args:
        image (np.array):  and input image in BGR format 
        orig_rgb (tuple): origin color in RGB format like [255,0,0]
        target_rgb (tuple): target color in RGB format like [0,255,0]
        mask (np.array) :  The mask to be used to change the color
    Returns:
        (np.array): image with the color changed in BGR format
        """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    orig_hsv = cv2.cvtColor(np.uint8([[orig_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    target_hsv = cv2.cvtColor(np.uint8([[target_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    delta_h = (int(target_hsv[0]) - int(orig_hsv[0])) % 180
    if delta_h > 90:
        delta_h -= 180

    delta_s = int(target_hsv[1]) - int(orig_hsv[1])
    delta_v = int(target_hsv[2]) - int(orig_hsv[2])
    
    h, s, v = cv2.split(hsv)
    h = np.mod(h.astype(np.int16) + delta_h, 180).astype(np.uint8)
    s = np.clip(s.astype(np.int16) + delta_s, 0, 255).astype(np.uint8)
    v = np.clip(v.astype(np.int16) + delta_v, 0, 255).astype(np.uint8)

    
    hsv_shifted = cv2.merge([h, s, v])
    bgr_shifted = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)
    
    result = cv2.bitwise_or(
        cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask)),
        cv2.bitwise_and(bgr_shifted, bgr_shifted, mask=mask)
    )
    return result

def get_color_on_video(in_path, color, lower, upper, tuning=25, fraction= 1, out_path=None):

    cap = cv2.VideoCapture(in_path)
    if not out_path:
        file_name = os.path.basename(in_path)[:-4]
        out_path = f"../output/color.mp4"

    #------------Set up------------#

    if not cap.isOpened():
        raise IOError(f"Could not open {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * fraction)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for frame_idx in range(total_frames):

        ok, frame = cap.read()
        if not ok:
            break
        
        print(f"Progress: {(frame_idx)/(total_frames) *100:.2f} %", end="\r")

        green_mask = detect_color(frame, color, lower, upper, tuning)
        green_pixels = cv2.bitwise_and(frame, frame, mask=green_mask)

        writer.write(green_pixels)

    cap.release()
    writer.release()

    return
