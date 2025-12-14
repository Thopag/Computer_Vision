import numpy as np

grp = 11

FIXED = True

if FIXED:
    t = "fixed"
else:
    t = "dynamic"

IN_PATH = f"input/group_{grp}/video_group_{grp}_{t}.mp4"
OUT_PATH = f"output/video_group_{grp}_{t}.mp4"
JSON_PATH = f"input/annotations_group_{grp}.json"
FILE_NAME = f"group_{grp}_{t}"

N_OBJECT = 3
# 0 : ball || 1 : bottle || 2 : obj3

# -------------- YOLO -------------- #

MODEL_PATH = "model/yolov8n.pt"
READ_EVERY_X_FRAME = 1
CONF_TRESHOLD = 0.0

BLACKLIST =['person', 'skateboard', 'laptop', 'cup', 'chair', 'dining table', 'microwave', 'umbrella', 
    'kite', 'cat', 'traffic light', 'book', 'cell phone', 'keyboard', 'scissors', 'frisbee', 'suitcase', 'dog', 'tv', "handbag"]



# -------------- TRACKER -------------- #

OBJ_TRACKEREND_FRAME = None

WAND_TRACKER_END_FRAME = None
WAND_TRACKER_START_FRAME = 1170 #1615
WAND_MAX_PRED_JUMP = 30 # was 30 for gr11

MIN_W = 10
MIN_H = 10

# -- Kalman filter -- #

KALMAN_OBJ_TIMER = 15

KALMAN_WAND_TIMER = 15
KF_WAND_PROCESS_NOISE = 1.0
KF_WAND_MEASUREMENT_NOISE = 2.0

# -- interpolation -- #

INTERPOLATION_TYPE = 'linear'

# -- blob -- #

BLOB_MIN_AREA = 8
BLOB_MAX_AREA = 300
MAX_BLOB_WIDTH = 45
MAX_BLOB_HEIGHT = 45

# -------------- COLOR -------------- #

# -- green -- #

GREEN = [0,255,0]
LOWER_GREEN = [40,40]
UPPER_GREEN = [255,255]
TUNNING_GREEN = 35

# -- red -- #
LOWER_RED1 = np.array([0, 120, 60])
UPPER_RED1 = np.array([10, 255, 255])

LOWER_RED2 = np.array([170, 120, 60])
UPPER_RED2 = np.array([180, 255, 255])

RED_MEDIAN_BLUR = 5

# -------------- TRICK 1 -------------- #

KERNEL_FRACTION = 0.75
WITH_FIRST_FRAME = False

S_DIL_CLOAK = 20
S_ERODE_ROI = 15
S_DIL_ROI = 15

# -------------- TRICK 2 -------------- #
WAND_ITER   = 3
WAND_DILATE = 20

OBJ_ITER    = 3
OBJ_DILATE  = 20