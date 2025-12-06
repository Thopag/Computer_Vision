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

    ball_frames = []
    bottle_frames = []
    mushroom_frames = []

    with open(path, "r") as f:
        next(f)
        for line in f:
            if ":" not in line:
                continue
            frame, txt = line.strip().split(":")
            frame = int(frame)
            txt = txt.lower()

            if "ball" in txt:
                ball_frames.append(frame)
            elif "bottle" in txt:
                bottle_frames.append(frame)
            elif "mushroom" in txt:
                mushroom_frames.append(frame)

    return {
        "ball": sorted(ball_frames),
        "bottle": sorted(bottle_frames),
        "mushroom": sorted(mushroom_frames)
    }
    
    

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