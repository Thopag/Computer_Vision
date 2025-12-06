import numpy as np 

# ===========================================================
# GROUP consecutive frames into blocks
# ===========================================================
def group_blocks(frames):
    if not frames:
        return []
    blocks = []
    curr = [frames[0]]
    for f in frames[1:]:
        if f == curr[-1] + 1:
            curr.append(f)
        else:
            blocks.append(curr)
            curr = [f]
    blocks.append(curr)
    return blocks


# ===========================================================
# SIMPLE BOX MASK
# ===========================================================
def box_to_mask(frame, cx, cy, w, h):
    H, W = frame.shape[:2]
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)

    x1 = max(0, min(W-1, x1))
    x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1))
    y2 = max(0, min(H-1, y2))

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask

