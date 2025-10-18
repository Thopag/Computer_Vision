import cv2
import numpy as np

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
    only_1_label = np.all(overlap_label == overlap_label)

    if only_1_label:
        componant_pixels = labels == overlap_label
        overlap_componant_mask = np.zeros_like(mask1)
        overlap_componant_mask[componant_pixels] = 255

        return True, overlap_componant_mask
    
    return False, None


# Version that does not work for all case, keep because the ordering of process
# this could give idea for futur optimisation
def get_overlap_componant_V0(mask1, mask2):

    intersection =  cv2.bitwise_and(mask1, mask1, mask=mask2).astype(np.uint8)

    are_diff = (intersection != mask2).any()
    if are_diff:
        return False, None

    num_labels, labels = cv2.connectedComponents(intersection)

    if num_labels == 1 or num_labels > 2:
        return False, None

    overlap_labels = labels[intersection != 0]
    overlap_label = overlap_labels[0]

    componant_pixels = labels == overlap_label
    overlap_componant_mask = np.zeros_like(mask1)
    overlap_componant_mask[componant_pixels] = 255

    return True, overlap_componant_mask


    