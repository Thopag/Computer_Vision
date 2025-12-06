import cv2
import numpy as np

class WandDetector:

    def __init__(self):
        # --- RED TIP ---
        self.lower_red1 = np.array([0, 120, 80])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 80])
        self.upper_red2 = np.array([180, 255, 255])

        # --- YELLOW DECORATION ---
        self.lower_yellow = np.array([15, 120, 80])
        self.upper_yellow = np.array([30, 255, 255])

        # --- BLACK SHAFT ---
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 60])

    # 1) DETECT WAND ANYWHERE (robust HSV fusion)
   
    def detect_wand_hsv(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Masks
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.lower_red1, self.upper_red1),
            cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        )
        yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        black_mask = cv2.inRange(hsv, self.lower_black, self.upper_black)

        # Combine → wand signature
        wand_mask = cv2.bitwise_or(red_mask, yellow_mask)
        wand_mask = cv2.bitwise_or(wand_mask, black_mask)

        # Clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        wand_mask = cv2.morphologyEx(wand_mask, cv2.MORPH_OPEN, kernel)
        wand_mask = cv2.dilate(wand_mask, kernel, iterations=2)

        cnts,_ = cv2.findContours(wand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        cnt = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)

        # compute a "wand confidence"
        red_pixels = np.count_nonzero(red_mask)
        yellow_pixels = np.count_nonzero(yellow_mask)
        black_pixels = np.count_nonzero(black_mask)

        wand_score = red_pixels*1.5 + yellow_pixels*1.2 + black_pixels*0.8

        return {
            "bbox": (x, y, x+w, y+h),
            "mask": wand_mask,
            "score": wand_score,
            "submasks": {
                "red": red_mask,
                "yellow": yellow_mask,
                "black": black_mask
            }
        }

    # 2) FIND TIP FROM THE HSV BOX
   
    def detect_tip(self, frame, bbox):

        x1,y1,x2,y2 = bbox
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.lower_red1, self.upper_red1),
            cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        )

        # Clean tip mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        cnts,_ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

        return (x1 + x + w//2, y1 + y + h//2)
