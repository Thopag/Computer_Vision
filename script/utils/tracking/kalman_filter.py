import numpy as np
import cv2
from script.CONFIG import *

# The Kalman filter has a timer, which is the number of frames before it enters sleep mode, 
# allowing control over when to stop trusting the Kalman filter

class KalmanBoxTracker:
    def __init__(self, init_box, time_before_sleep):
        
        self.time_before_sleep = time_before_sleep
        self.timer = self.time_before_sleep
        self.last_pred = None

        self.kf = cv2.KalmanFilter(8, 4)
        dt = 1.0

        # State transition: [cx cy w h vx vy vw vh]
        self.kf.transitionMatrix = np.array([
            [1,0,0,0, dt,0,0,0],
            [0,1,0,0, 0,dt,0,0],
            [0,0,1,0, 0,0,dt,0],
            [0,0,0,1, 0,0,0,dt],
            [0,0,0,0, 1,0,0,0],
            [0,0,0,0, 0,1,0,0],
            [0,0,0,0, 0,0,1,0],
            [0,0,0,0, 0,0,0,1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * KF_OBJ_PROCESS_NOISE
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * KF_OBJ_MEASUREMENT_NOISE

        cx,cy,w,h = init_box
        self.kf.statePost = np.array([[cx],[cy],[w],[h],[0],[0],[0],[0]], dtype=np.float32)

        # Make a first prediction to store a value in last_pred
        self.predict()

    def predict(self):
        self.last_pred = self.kf.predict()[:4].ravel()
        return self.last_pred

    def update(self, det):
        meas = np.array(det, dtype=np.float32).reshape(4,1)
        self.kf.correct(meas)
        self.timer = self.time_before_sleep

    def decrement_timer(self):
        self.timer = self.timer - 1
    
    def is_active(self):
        return self.timer > 0

class KalmanWandTracker:
    def __init__(self, init_position, time_before_sleep):

        self.time_before_sleep = time_before_sleep
        self.timer = self.time_before_sleep

        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0

        self.kf.transitionMatrix = np.array([
            [1,0,dt,0],
            [0,1,0,dt],
            [0,0,1,0],
            [0,0,0,1]
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.eye(2,4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * KF_WAND_PROCESS_NOISE
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KF_WAND_MEASUREMENT_NOISE

        self.kf.statePost = np.array([[init_position[0]],[init_position[1]],[0],[0]], dtype=np.float32)

        # Make a first prediction to define self.px and self.py
        pred = self.kf.predict()
        self.px = pred[0]
        self.py = pred[1]

    def predict(self):
        px, py = self.kf.predict()[:2].ravel()

        px = np.clip(px, self.px-WAND_MAX_PRED_JUMP, 
                            self.px+WAND_MAX_PRED_JUMP) 
        py = np.clip(py, self.py-WAND_MAX_PRED_JUMP,
                            self.py+WAND_MAX_PRED_JUMP)

        self.px = px
        self.py = py
        return px, py

    def update(self, det):
        cx, cy = det["center"]
        meas = np.array([[cx],[cy]], dtype=np.float32)
        self.kf.correct(meas)
        self.timer = self.time_before_sleep

    def decrement_timer(self):
        self.timer = self.timer - 1
    
    def is_active(self):
        return self.timer > 0
