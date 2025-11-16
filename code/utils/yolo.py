import cv2
import numpy as np
from ultralytics import YOLO

def detect_objects(frame , model):
    """
    Detect objects in an image using YOLOv8 and convert boxes to masks.
    
    Args:
        frame: BGR image (numpy array)
    
    Returns:
        masks: list of uint8 masks (1 inside box, 0 outside)
        labels: list of class names corresponding to each mask
    """
    # Convert image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(image_rgb)[0]

    masks = []
    labels = []

    H, W = frame.shape[:2]
    class_names = results.names

    # Process detections
    boxes = results.boxes
    for box in boxes:
        # Box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Create mask for this object
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=1, thickness=-1)
        masks.append(mask)

        # Get class label
        class_id = int(box.cls[0])
        class_name = class_names[class_id]
        labels.append(class_name)

    return masks, labels