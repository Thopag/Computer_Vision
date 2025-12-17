import numpy as np

grp = 11

FIXED = True

if FIXED:
    t = "fixed"
else:
    t = "dynamic"

IN_PATH = f"input/group_{grp}/video_group_{grp}_{t}.mp4"
OUT_PATH = f"output/video_group_{grp}_{t}.mp4"
JSON_PATH = f"input/group_{grp}/annotations_group_{grp}.json"
FILE_NAME = f"group_{grp}_{t}"

# -------------- YOLO -------------- #

MODEL_PATH = "model/yolov8n.pt"
READ_EVERY_X_FRAME = 1
CONF_TRESHOLD = 0.0

BLACKLIST = ['person', 'dining table', 'sink', 'tie', 'cup', 'remote', 'skateboard', 
        'orange', 'handbag', 'cake', 'bowl', 'frisbee', 'umbrella', 'scissors', 'vase', 'book' , 'tennis racket']
CONTROL_LIST = []

# -------------- TRACKER -------------- #

N_OBJECT = 3
# 0 : ball || 1 : bottle || 2 : obj3

# -- Timing -- #
OBJ_TRACKER_START_FRAME = 570 # (19s)
OBJ_TRACKER_END_FRAME = 3120 # (104s)

OBJ_BALL_TRACKER_START_FRAME = 3270 # (109s)
OBJ_BALL_TRACKER_END_FRAME = None

WAND_2_TRACKER_START_FRAME = 2310 # (77s)
WAND_2_TRACKER_END_FRAME = 3120 # (104s)


# -- prediction -- #
MAX_W_OBJ = 75
MAX_H_OBJ = 175

ALPHA_CIOU = None



MIN_W = 10
MIN_H = 10

# -- Kalman filter -- #


KF_OBJ_PROCESS_NOISE = 0.001
KF_OBJ_MEASUREMENT_NOISE = 0.05



# -- interpolation -- #

INTERPOLATION_TYPE = 'linear'
#INTERPOLATION_TYPE_trick3 = 'cubic'


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

KALMAN_OBJ_TIMER = 5
KF_OBJ_PROCESS_NOISE = 0.001
KF_OBJ_MEASUREMENT_NOISE = 0.05
 #------wand------------------
KALMAN_WAND_TIMER = 10
KF_WAND_PROCESS_NOISE = 1.0
KF_WAND_MEASUREMENT_NOISE = 0.01
WAND_MAX_PRED_JUMP = 29
KALMAN_WAND_TIMER = 20

# -- red -- #  WAND
LOWER_RED1 = np.array([160, 90, 40])
UPPER_RED1 = np.array([180, 255, 255])

LOWER_RED2 = np.array([170, 90, 40])
UPPER_RED2 = np.array([180, 255, 255])

RED_MEDIAN_BLUR = 5

WAND_3_TRACKER_START_FRAME = 3600 # (124s)
WAND_3_TRACKER_END_FRAME = None



# Pixel at (472,620)
#   BGR = (97, 94, 168)
#   HSV = (179, 112, 168)
# ------------------------------
# Pixel at (471,625)
#   BGR = (24, 20, 90)
#   HSV = (178, 198, 90)
# ------------------------------    
# Pixel at (474,617)
#   BGR = (98, 96, 161)
#   HSV = (179, 103, 161)

# -------------- TRICK 1 -------------- #

KERNEL_FRACTION = 0.75
WITH_FIRST_FRAME = False

S_DIL_CLOAK = 20
S_ERODE_ROI = 15
S_DIL_ROI = 20

# -------------- TRICK 2 -------------- #

WAND_ITER   = 3
WAND_DILATE = 20

OBJ_ITER    = 3
OBJ_DILATE  = 20

# -------------- TRICK 3 -------------- #

TRICK3_ALPHA = 0.7
TRICK3_AMP_X = 2.0

TRICK3_AMP_Y_UP   = 0.2   # wand vers le haut -> balle monte très peu
TRICK3_AMP_Y_DOWN = 1.2   # wand vers le bas -> balle descend beaucoup

TRICK3_MAX_DX = 20
TRICK3_MAX_DY = 15

TRICK3_MAX_MISSING_SEC= 1
TRICK3_MISSING_DECAY= 0.95   # amortissement des absences de tracking