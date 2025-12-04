import cv2
from ultralytics import YOLO

class YOLODetector:

    def __init__(self, model_path="yolov8s.pt"):
        print(f"[YOLO] Loading model: {model_path}")
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Applique YOLOv8 au frame.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.model(rgb, verbose=False)[0]
        class_names = results.names

        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = class_names[cls_id]

            detections.append((cls_name, x1, y1, x2, y2, conf))

        return detections
