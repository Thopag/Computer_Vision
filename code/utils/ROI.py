import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLO model once (do this outside the function)
model = YOLO('yolov8n.pt')

# Generate consistent colors for classes
np.random.seed(42)
colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

def is_some_overlap(mask1, mask2):
    """
    Check if there is at least 1 pixel in mask1 and mask2 that overlaps
    """

    return cv2.bitwise_and(mask1, mask1, mask=mask2).any()

def get_overlap_componant(cover_mask, covered_mask):
    """
    Return the mask of the componant (1 only) in cover_mask that fully cover the 
    covered_mask if it exist.

    cover_mask: mask where we look after the componant
    covered_mask: mask of the element that need to be overlapped

    return
    bool: True if there is overlapping, False otherwise
    mask: with the overlapping componant

    """
    cover_mask = cover_mask.astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(cover_mask)

    # Check if not only background
    if num_labels == 1:
        return False, labels

    intersection =  cv2.bitwise_and(labels, labels, mask=covered_mask)

    no_zero_inter = intersection != 0

    # verif if intersection = covered_mask
    are_diff = ( no_zero_inter != (covered_mask != 0)).any()
    if are_diff:
        return False, labels

    overlap_labels = intersection[no_zero_inter]

    # Check if it is not only the background
    if len(overlap_labels) == 0:
        return False, labels

    overlap_label = overlap_labels[0]
    only_1_label = np.all(overlap_labels == overlap_label)

    if only_1_label:
        componant_pixels = labels == overlap_label
        overlap_componant_mask = np.zeros_like(cover_mask)
        overlap_componant_mask[componant_pixels] = 255

        return True, overlap_componant_mask

    return False, labels

def roi(mask, s_erod, s_dil):
    """
    # could add directly the mask as input
    Create a region of interest mask based

    s_erod: size for erosion
    s_dil: size for dilation
    mask : input mask

    return 
        mask of format 
    """

    # we erode first to remove noise (small white regions detected by mistake)
    kernel_erod = (s_erod , s_erod)
    SE= cv2.getStructuringElement(cv2.MORPH_RECT,kernel_erod)
    eroded_mask = cv2.erode(mask,SE)

    #then we dilate to get the full region of interest around the detected object

    kernel_dil = (s_dil , s_dil)
    SE= cv2.getStructuringElement(cv2.MORPH_RECT,kernel_dil)

    roi_mask = cv2.dilate(eroded_mask,SE)
    # Change here
    ok = not np.all(roi_mask == 0)

    return ok, roi_mask

def detect_objects(frame, confidence_threshold=0):
    """
    Detect objects in an image using YOLOv8 and convert boxes to masks.

    Args:
        frame: BGR image (numpy array)

    Returns:
        masks: list of uint8 masks (1 inside box, 0 outside)
        labels: list of class names corresponding to each mask
    """
    # Load YOLO model
    #model = YOLO('yolov8n.pt')

    # Convert image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(image_rgb, verbose=False)[0]

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

def annotate_frame(frame, confidence_threshold=0.0, file=None):
    """
    Detect objects on a single frame and draw bounding boxes.

    Args:
        frame (np.ndarray): Input image/frame in BGR format
        confidence_threshold (float): Minimum confidence to draw box

    Returns:
        np.ndarray: Annotated frame with bounding boxes
    """
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

        if file:
            file.write(f"[{class_names[class_id]}: {conf:.2f}], " )

    if file:
        file.write(f"\n" )

    return annotated_frame