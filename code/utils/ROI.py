import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_overlap_componant(mask1, mask2):

    mask1 = mask1.astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(mask1)
    if num_labels == 1:
        return False, None
    
    intersection =  cv2.bitwise_and(labels, labels, mask=mask2)

    no_zero_inter = intersection != 0
    are_diff = ( no_zero_inter != (mask2 != 0)).any()
    if are_diff:
        return False, None

    overlap_labels = intersection[no_zero_inter]

    if len(overlap_labels) == 0:
        return False, None
    
    overlap_label = overlap_labels[0]
    only_1_label = np.all(overlap_labels == overlap_label)

    if only_1_label:
        componant_pixels = labels == overlap_label
        overlap_componant_mask = np.zeros_like(mask1)
        overlap_componant_mask[componant_pixels] = 255

        return True, overlap_componant_mask
    
    return False, None

def roi(img, s_erod, s_dil, mask):
    """
    # could add directly the mask as input
    Create a region of interest mask based

    s_erod: size for erosion
    s_dil: size for dilation
    img : input image
    
    return 
        image of BGR format 
    """

    # we erode first to remove noise (small white regions detected by mistake)
    kernel_erod = (s_erod , s_erod)
    SE= cv2.getStructuringElement(cv2.MORPH_RECT,kernel_erod)
    eroded_mask = cv2.erode(mask,SE)
    
    #then we dilate to get the full region of interest around the detected object
    
    kernel_dil = (s_dil , s_dil)
    SE= cv2.getStructuringElement(cv2.MORPH_RECT,kernel_dil)
    
    roi_mask = cv2.dilate(eroded_mask,SE)
    
    new_image = cv2.bitwise_and(img , img , mask = roi_mask)
    
    #for visualization
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    return [new_image, roi_mask]
