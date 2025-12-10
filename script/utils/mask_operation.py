import cv2
import numpy as np
import matplotlib.pyplot as plt
from script.CONFIG import *

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

    return roi_mask

def extract_blobs(mask):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []

    for c in contours:
        area = cv2.contourArea(c)
        if not (BLOB_MIN_AREA <= area <= BLOB_MAX_AREA):
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w > MAX_BLOB_WIDTH or h > MAX_BLOB_HEIGHT:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])

        blobs.append({"center": (cx,cy), "bbox": (x,y,w,h), "area": area})

    return blobs