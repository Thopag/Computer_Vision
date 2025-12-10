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
# LOAD ready.txt
# ===========================================================
def load_ready(path):
    data = {}
    with open(path, "r") as f:
        next(f)
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
# LOAD interactions.txt
# ===========================================================
def load_interactions(path):

    inter = {0: [], 1: [], 2: []}

    with open(path, "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            frame = int(parts[0])
            obj_id = int(parts[-1])

            if obj_id in inter:
                inter[obj_id].append(frame)

    # Sort
    for k in inter:
        inter[k] = sorted(inter[k])

    # =======================================================
    # APPLY FILTERING **ONLY TO BOTTLE (id = 1)**
    # =======================================================
    bottle_frames = inter[1]
    bottle_blocks = group_blocks(bottle_frames)

    if len(bottle_blocks) > 2:
        # Keep only FIRST and LAST block
        first_blk = bottle_blocks[0]
        last_blk  = bottle_blocks[-1]
        inter[1] = first_blk + last_blk
    else:
        # Keep as is
        inter[1] = bottle_frames

    # Everything else untouched:
    # - Ball (0) → keep original
    # - Object3 (2) → keep original

    return inter


# ============================================================
# LOAD WAND LOG (smoothed KF)
# ============================================================
def load_wand_log(path):
    wand = {}
    with open(path, "r") as f:
        next(f)
        for line in f:
            frame, tid, cx, cy, w, h = line.strip().split(",")
            frame = int(frame)
            cx, cy, w, h = map(float, (cx, cy, w, h))
            wand[frame] = {"cx": cx, "cy": cy, "w": w, "h": h}
    return wand