import cv2
import numpy as np

def detect_color(image, color, tuning=25):
    """
    Detect a specific color in an image and create a mask.

    Args:
        image (_type_): and input image in BGR format
        color (_type_): a list RGB color like [255,0,0]
        tuning (int, optional):tuning parameter. Defaults to 25.

    Returns:
        _type_: _description_
        
    """
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color = np.uint8([[color]])
    
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_color)
    H = h[0][0]
    lower_h = max(H - tuning, 0)
    upper_h = min(H + tuning, 179)
    lower = np.array([lower_h, 100, 100])
    upper = np.array([upper_h, 255, 255])
    
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def change_color(image, orig_rgb, target_rgb, tuning=25):
    """
    Change a specific color in an image to a target color.

    Args:
        image (_type_): _description_
        orig_rgb (_type_): origin color in RGB format like [255,0,0]
        target_rgb (_type_): target color in RGB format like [0,255,0]
        tuning (int, optional): _description_. Defaults to 25.

    Returns:
        _type_: image with the color changed in BGR format
        """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = detect_color(image, orig_rgb, tuning)
    
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

   
