import cv2
import numpy as np

class BallDetector:
    def __init__(self):
        # HSV range tuned for blue ball
        self.lower = np.array([90, 80, 50])
        self.upper = np.array([130, 255, 255])

    def detect(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # find contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, None, None

        # largest contour = the ball
        cnt = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        cx = x + w // 2
        cy = y + h // 2

        return (cx, cy), mask, (x, y, x+w, y+h)
