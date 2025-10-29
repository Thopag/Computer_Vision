import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_overlap_componant(cover_mask, covered_mask):

    cover_mask = cover_mask.astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(cover_mask)
    if num_labels == 1:
        return False, labels
    
    intersection =  cv2.bitwise_and(labels, labels, mask=covered_mask)

    no_zero_inter = intersection != 0
    are_diff = ( no_zero_inter != (covered_mask != 0)).any()
    if are_diff:
        return False, labels

    overlap_labels = intersection[no_zero_inter]

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

    return roi_mask
