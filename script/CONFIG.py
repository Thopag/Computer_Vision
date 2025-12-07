
grp = 11

FIXED = False

if FIXED:
    t = "fixed"
else:
    t = "dynamic"

IN_PATH = f"input/video_group_{grp}_{t}.mp4"
OUT_PATH = f"output/video_group_{grp}_{t}.mp4"
JSON_PATH = f"input/annotations_group_{grp}.json"
FILE_NAME = f"group_{grp}_{t}"

N_OBJECT = 3

# -------------- COLOR -------------- #

GREEN = [0,255,0]
LOWER_GREEN = [40,40]
UPPER_GREEN = [255,255]
TUNNING_GREEN = 35

# -------------- TRICK 1 -------------- #

KERNEL_FRACTION = 0.75
WITH_FIRST_FRAME = False

S_DIL_CLOAK = 20
S_ERODE_ROI = 15
S_DIL_ROI = 15