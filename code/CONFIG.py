import numpy as np

IN_PATH = 404
OUT_PATH =  404
FILE_NAME = "group_11_dynamic"


#obj["id"] == 0:   # bottle  obj["id"] == 1: # Object3    obj["id"] == 2: # ball
CONFIG = {
    "IN_PATH": "input/dynamic/trick2.mp4",
    "OUT_PATH": "output/object_tracker.mp4",
    "OUT_PATH_Wand": "output/wand_tracker.mp4",
    "LOG_PATH": "info/trick2_log.txt",
    "LOG_PATH_Wand":"info/wand_tracker.txt",
    "VIDEO_READY" : "output/trick2_visual.mp4",
    "FINAL"       : "info/object_positions.txt",
    "INTERACTION" : "info/interactions.txt",
    "START_FRAME": 150,
    "CONF_THR": 0.25,
    "IOU_THR": -np.inf,
    "MODEL": "code/models/yolov8n.pt",
    "N_OBJECTS": 3,
    "nbr_frame_before_sleep": 30,
    "blacklist" : ['person', 'skateboard', 'laptop', 'cup', 'chair', 'dining table', 'microwave', 'umbrella', 
    'kite', 'cat', 'traffic light', 'book', 'cell phone', 'keyboard', 'scissors', 'frisbee', 'suitcase', 'dog', 'tv'],
    # --- Strict red detector ---
    # Higher = require stronger red → reduces false positives
    # Lower  = detect more red → useful if lighting is dark
    "red_min_s": 120,
    "red_min_v": 60,
    "median_blur": 5,      # remove tiny noise dots

    # --- Red blob filtering ---
    # Only accept blobs that look like the wand tip
    "blob_min_area": 8,     # ignore noise
    "blob_max_area": 300,   # ignore large red regions (e.g. mushroom)
    "max_blob_width": 45,   # wand tip is small
    "max_blob_height": 45,

    # --- Kalman filter tuning ---
    # process_noise:
    #     higher  = smoother, slower movement
    #     lower   = more reactive, less smooth
    "kf_process_noise": 0.1,

    # measurement_noise:
    #     higher  = trust detection less (more smoothing)
    #     lower   = trust detection more (faster reactions)
    "kf_measurement_noise": 5.0,

    # --- Limit impossible jumps (px/frame) ---
    # Prevents teleporting to mushroom or bottle
    "max_prediction_jump": 40,

    # --- Backward smoothing (after full tracking) ---
    # If a single frame jumps too far → correct it
    "smooth_jump_threshold": 45,
    "smooth_window": 2,
    "max_frame_gap": 12,   # do NOT smooth when wand is out of frame too long

    # --- Output bounding box size (in log file) ---
    "bbox_size": 16
}
