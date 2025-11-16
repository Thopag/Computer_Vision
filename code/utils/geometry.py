import cv2
import numpy as np 

def grow_object(frame, mask_obj, scale=3.0):
    # 1. Find bounding box of the object
    ys, xs = np.where(mask_obj > 0)
    if len(xs) == 0 or len(ys) == 0:
        return frame  # nothing to enlarge
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # 2. Extract the object
    object_crop = frame[y_min:y_max+1, x_min:x_max+1]
    mask_crop   = mask_obj[y_min:y_max+1, x_min:x_max+1]

    # 3. Resize both object and mask
    new_w = int(object_crop.shape[1] * scale)
    new_h = int(object_crop.shape[0] * scale)

    object_big = cv2.resize(object_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_big   = cv2.resize(mask_crop,   (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 4. Compute placement (center on original object center)
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    x0 = center_x - new_w // 2
    y0 = center_y - new_h // 2
    x1 = x0 + new_w
    y1 = y0 + new_h

    # 5. Clip to stay in frame
    h, w = frame.shape[:2]
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)

    # Regions in the enlarged object
    crop_x0 = x0c - x0
    crop_y0 = y0c - y0
    crop_x1 = crop_x0 + (x1c - x0c)
    crop_y1 = crop_y0 + (y1c - y0c)

    # 6. Paste enlarged object
    frame_out = frame.copy()
    roi = frame_out[y0c:y1c, x0c:x1c]

    alpha = (mask_big[crop_y0:crop_y1, crop_x0:crop_x1] > 0).astype(np.uint8)

    # Replace pixels where mask is 1
    roi[alpha == 1] = object_big[crop_y0:crop_y1, crop_x0:crop_x1][alpha == 1]

    frame_out[y0c:y1c, x0c:x1c] = roi

    return frame_out