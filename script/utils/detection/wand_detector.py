from script.CONFIG import *
from script.utils.tracking.trajectory import object_trajectory
from script.utils.box import cxcywh_to_xyxy
from script.utils.detection.color import detect_red_strict
from script.utils.mask_operation import extract_blobs

def remove_objects(frame, box_list):
    """
    Paints object bounding boxes black (removes them).
    So wand-detector sees a frame without the object colors.
    (like our red mushroom)

    Args:
        frame: The frame to clean
        box_list: The box of the object to remove

    Return:
        The cleanned frame
    """
    if box_list is None:
        return frame

    clean = frame.copy()

    for obj_box in box_list:

        if obj_box:
            x1, y1, x2, y2 = cxcywh_to_xyxy(obj_box)
            clean[y1:y2, x1:x2] = 0

    return clean

class wand_detector:
    """
    Class to extract the blobs of a given frame
    """

    def __init__(self, obj_trajs : object_trajectory):

        self.obj_trajs = obj_trajs
        self.blobs = []

    def detect(self, frame, frame_idx):

        obj_boxes = self.obj_trajs.boxs_at_frame(frame_idx)
        
        clean_frame = remove_objects(frame, obj_boxes)
        red_mask = detect_red_strict(clean_frame)
        blobs = extract_blobs(red_mask)

        self.blobs = blobs
        return blobs

    def have_something(self):

        if len(self.blobs) == 0:
            return False

        return True