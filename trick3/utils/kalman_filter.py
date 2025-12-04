import numpy as np

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
