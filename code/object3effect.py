import cv2
import numpy as np


# ===========================================================
# CONTOUR MASK FROM BOUNDING BOX
# ===========================================================
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


# ===========================================================
# PLACE LOCAL MASK INTO FULL FRAME
# ===========================================================
def place_mask_on_frame(mask_crop, frame_shape, cx, cy, w, h):

    H, W = frame_shape[:2]

    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)

    x1 = max(0, min(W-1, x1))
    x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1))
    y2 = max(0, min(H-1, y2))

    full_mask = np.zeros((H, W), dtype=np.uint8)

    if mask_crop is None:
        return full_mask

    full_mask[y1:y2, x1:x2] = mask_crop

    return full_mask


# ===========================================================
# APPLY SELECTIVE BLUR USING MASK
# ===========================================================
def selective_blur_with_mask(frame, full_mask):

    blurred = cv2.GaussianBlur(frame, (35, 35), 0)

    mask_3d = cv2.merge([full_mask, full_mask, full_mask]) / 255.0

    result = frame * mask_3d + blurred * (1 - mask_3d)
    result = result.astype(np.uint8)

    return result


# ===========================================================
# LOAD OBJECT POSITIONS FILE
# ===========================================================
def load_ready(path):

    data = {}

    with open(path, "r") as f:
        next(f)  # skip header
        for line in f:
            frame, tid, cx, cy, w, h = line.strip().split(",")

            frame = int(frame)
            tid   = int(tid)
            cx, cy, w, h = map(float, (cx, cy, w, h))

            if frame not in data:
                data[frame] = []

            data[frame].append({
                "id": tid,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h
            })
    return data


# ===========================================================
# RUN DOF BLUR EFFECT + SAVE VIDEO
# ===========================================================
def run_blur_effect(video_path, ready_path, save_path, target_id=2):

    detections = load_ready(ready_path)
    cap = cv2.VideoCapture(video_path)

    # Video writer setup
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # If detections exist for this frame
        if frame_idx in detections:
            for obj in detections[frame_idx]:
                if obj["id"] == target_id:

                    # 1. Local contour mask
                    mask_crop = find_contour_from_bbox(
                        frame,
                        obj["cx"], obj["cy"],
                        obj["w"], obj["h"]
                    )

                    # 2. Full-frame mask
                    full_mask = place_mask_on_frame(mask_crop, frame.shape,
                                                    obj["cx"], obj["cy"],
                                                    obj["w"], obj["h"])

                    # 3. Apply selective blur
                    frame = selective_blur_with_mask(frame, full_mask)

        out.write(frame)

        cv2.imshow("Selective Blur DOF", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nSaved blurred video to: {save_path}")


# ===========================================================
# RUN
# ===========================================================
run_blur_effect(
    video_path="input/dynamic/trick2.mp4",
    ready_path="info/object_positions.txt",
    save_path="output/blurred_result.mp4",
    target_id=1
)
