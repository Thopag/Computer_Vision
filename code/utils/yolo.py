import cv2
import os
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load the model ONCE

# Generate consistent colors for classes
np.random.seed(42)
colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

def detect_objects(frame, confidence_threshold=0.0):
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
    results = model(image_rgb , verbose= False)[0]

    masks = []
    labels = []

    H, W = frame.shape[:2]
    class_names = results.names

    # Process detections
    boxes = results.boxes
    for box in boxes:
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue
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

def annotate_frame(in_path, out_path=None, confidence_threshold=0.0):
    """
    Detect objects on a single frame and draw bounding boxes.

    Args:
        frame (np.ndarray): Input image/frame in BGR format
        confidence_threshold (float): Minimum confidence to draw box

    Returns:
        np.ndarray: Annotated frame with bounding boxes CHANANNANANA
    """

    cap = cv2.VideoCapture(in_path)
    if not out_path:
        file_name = os.path.basename(in_path)
        out_path = f"../yolo_output/{file_name}.mp4"

    txt_path = out_path[:-4] + ".txt"

    f = open(txt_path, "w", encoding="utf-8")

    #------------General set up------------#

    if not cap.isOpened():
        raise IOError(f"Could not open {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # YOLO expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(rgb_frame)[0]
    boxes = results.boxes
    class_names = results.names

    # Make a copy to draw on
    annotated_frame = frame.copy()

    # Draw boxes
    for box in boxes:
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = colors[class_id % len(colors)].tolist()
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, f"{class_names[class_id]} {conf:.2f}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated_frame