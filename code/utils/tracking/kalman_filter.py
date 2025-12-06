import numpy as np
import cv2

class KalmanFilter:
    def __init__(self):
        # state: x, y, vx, vy
        self.state = np.zeros((4,1))
        self.P = np.eye(4) * 1000  

        self.F = np.array([[1,0,1,0],
                           [0,1,0,1],
                           [0,0,1,0],
                           [0,0,0,1]], dtype=float)

        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=float)

        self.R = np.eye(2)*5
        self.Q = np.eye(4)*0.01

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        x, y = self.state[0,0], self.state[1,0]
        return x, y

    def update(self, meas_x, meas_y):
        z = np.array([[meas_x],[meas_y]])
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

class KalmanBoxTracker:
    def __init__(self, init_box, time_before_sleep):
        self.kf = cv2.KalmanFilter(8, 4)
        self.time_before_sleep = time_before_sleep
        self.timer = self.time_before_sleep
        self.last_pred = None

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
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.001
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.05

        cx,cy,w,h = init_box
        self.kf.statePost = np.array([[cx],[cy],[w],[h],[0],[0],[0],[0]], dtype=np.float32)

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
