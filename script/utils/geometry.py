import cv2
import numpy as np 


def find_contour_from_bbox(frame, cx, cy, w, h,
                           canny_low=15, canny_high=150,
                           blur_edges=3, dilate_iter=4, erode_iter=3,
                           min_area_ratio=0.001, max_area_ratio=0.95):

    H, W = frame.shape[:2]

    # Convert bbox to coordinates
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)

    x1 = max(0, min(W-1, x1))
    x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1))
    y2 = max(0, min(H-1, y2))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # 1) Grayscale, blur a bit for smoother edges
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_edges, blur_edges), 0)

    # 2) Canny edges
    edges = cv2.Canny(gray, canny_low, canny_high)

    # 3) Strengthen contours
    edges = cv2.dilate(edges, None, iterations=dilate_iter)
    edges = cv2.erode(edges, None, iterations=erode_iter)

    # 4) Contours extraction
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not cnts:
        return None

    crop_area = crop.shape[0] * crop.shape[1]
    min_area = crop_area * min_area_ratio
    max_area = crop_area * max_area_ratio

    # 5) Create empty mask
    mask = np.zeros(edges.shape, dtype=np.uint8)

    # Fuse all accepted contours
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            cv2.fillConvexPoly(mask, c, 255)

    # 6) Smooth mask slightly
    mask = cv2.GaussianBlur(mask, (9,9), 0)
    mask = (mask > 20).astype(np.uint8) * 255

    return mask

def grow_object_magic(frame, cx, cy, w, h,
                      frame_idx, first_touch, last_touch,
                      max_scale=3.0, feather=15, blur_amount=11):

    H, W = frame.shape[:2]

    # Original bounding box
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)

    x1 = max(0, min(W-1, x1))
    x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1))
    y2 = max(0, min(H-1, y2))

    # Extract contour mask inside bounding box
    contour_local = find_contour_from_bbox(frame, cx, cy, w, h)

    if contour_local is None:
        return frame

    # Place contour mask into global frame coordinates
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = contour_local

    if np.count_nonzero(mask) < 30:
        return frame

    # Compute scale progress
    if last_touch == first_touch:
        progress = 1
    else:
        progress = (frame_idx - first_touch) / (last_touch - first_touch)
        progress = np.clip(progress, 0, 1)

    scale = 1 + progress * (max_scale - 1)

    # Extract object region using mask
    ys, xs = np.where(mask > 0)
    y0, y1b = ys.min(), ys.max()
    x0, x1b = xs.min(), xs.max()

    obj = frame[y0:y1b+1, x0:x1b+1]
    obj_mask = mask[y0:y1b+1, x0:x1b+1]

    oh, ow = obj.shape[:2]
    new_h = int(oh * scale)
    new_w = int(ow * scale)

    obj_big = cv2.resize(obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_big = cv2.resize(obj_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Anchor bottom to table
    bottom_y = y1b  # original bottom pixel

    Y1 = bottom_y
    Y0 = Y1 - new_h

    center_x = (x0 + x1b) // 2
    X0 = center_x - new_w//2
    X1 = center_x + new_w//2

    X0c, Y0c = max(0,X0), max(0,Y0)
    X1c, Y1c = min(W,X1), min(H,Y1)

    ox0 = X0c - X0
    oy0 = Y0c - Y0
    ox1 = ox0 + (X1c - X0c)
    oy1 = oy0 + (Y1c - Y0c)


    # Feather alpha mask for smooth growth
    alpha = (mask_big > 0).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (feather, feather), 0)
    alpha = np.clip(alpha, 0, 1)

    alpha_crop = alpha[oy0:oy1, ox0:ox1]
    obj_region = obj_big[oy0:oy1, ox0:ox1]

    if blur_amount > 0:
        obj_region = cv2.GaussianBlur(obj_region, (blur_amount, blur_amount), 0)

    # Composite
    frame_out = frame.copy()
    roi = frame_out[Y0c:Y1c, X0c:X1c]
    alpha_exp = alpha_crop[...,None]

    roi = (roi*(1-alpha_exp) + obj_region*alpha_exp).astype(np.uint8)

    frame_out[Y0c:Y1c, X0c:X1c] = roi

    return frame_out

def fade_object_magic(frame, cx, cy, w, h,
                      frame_idx, fade_start, fade_end,
                      fade_in=False,
                      feather=15, blur_amount=7):

    H, W = frame.shape[:2]

    # Extract contour mask
    mask_local = find_contour_from_bbox(frame, cx, cy, w, h)
    if mask_local is None:
        return frame

    # Convert local mask to global coordinates
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x1 = max(0, min(W-1, x1))
    y1 = max(0, min(H-1, y1))

    mask = np.zeros((H, W), dtype=np.uint8)
    hL, wL = mask_local.shape
    mask[y1:y1+hL, x1:x1+wL] = mask_local

    if np.count_nonzero(mask) < 20:
        return frame

    # Compute fade progress
    if fade_end == fade_start:
        progress = 1
    else:
        progress = (frame_idx - fade_start) / (fade_end - fade_start)
        progress = np.clip(progress, 0, 1)

    # fade_in = True  -> alpha goes from 0 -> 1
    # fade_in = False -> alpha goes from 1 -> 0
    alpha = progress if fade_in else (1 - progress)

    # Object extraction
    ys, xs = np.where(mask > 0)
    y0, y1b = ys.min(), ys.max()
    x0, x1b = xs.min(), xs.max()

    obj = frame[y0:y1b+1, x0:x1b+1]
    obj_mask = mask[y0:y1b+1, x0:x1b+1]

    # feathered mask
    alpha_mask = cv2.GaussianBlur((obj_mask/255).astype(np.float32),
                                  (feather, feather), 0)
    alpha_mask = np.clip(alpha_mask, 0, 1) * alpha

    # optional blur
    obj_blur = cv2.GaussianBlur(obj, (blur_amount, blur_amount), 0)

    # Composite (fade to background)
    frame_out = frame.copy()
    roi = frame_out[y0:y1b+1, x0:x1b+1]

    alpha_exp = alpha_mask[..., None]

    roi = (obj_blur * alpha_exp + roi * (1 - alpha_exp)).astype(np.uint8)
    frame_out[y0:y1b+1, x0:x1b+1] = roi

    return frame_out
