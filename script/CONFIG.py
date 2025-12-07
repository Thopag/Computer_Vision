
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

# -------------- YOLO -------------- #

MODEL_PATH = "model/yolov8n.pt"
READ_EVERY_X_FRAME = 5
CONF_TRESHOLD = 0.0

BLACKLIST = ['person', 'skateboard', 'laptop', 'cup', 'chair', 'dining table', 'microwave', 'umbrella', 
    'kite', 'cat', 'traffic light', 'book', 'cell phone', 'keyboard', 'scissors', 'frisbee', 'suitcase', 'dog', 'tv', "handbag"]

# -------------- TRACKER -------------- #

KALMAN_TIMER = 15
END_FRAME = None

INTERPOLATION_TYPE = 'linear'

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