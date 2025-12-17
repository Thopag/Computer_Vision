import numpy as np
from ..box import box_to_mask

class object_trajectory:

    def __init__(self, path):
        
        with open(path, "r") as f:

            # Get the nbr of object written in the header
            _, value = next(f).strip().split(":", 1)
            self.nbr_object = int(value.strip())
            self.obj_dict =  {int(i): {"frame" : [], "cx" : [], "cy" : [], "w" : [], "h" : []} for i in range(self.nbr_object)}
            next(f)

            for line in f:
                frame, tid, cx, cy, w, h = line.strip().split(",")
                frame, tid = map(int, (frame, tid))
                cx, cy, w, h = map(float, (cx, cy, w, h))

                self.obj_dict[tid]["frame"].append(frame)
                self.obj_dict[tid]["cx"].append(cx)
                self.obj_dict[tid]["cy"].append(cy)
                self.obj_dict[tid]["w"].append(w)
                self.obj_dict[tid]["h"].append(h)
    
    def get_nbr_object(self):
        return self.nbr_object

    def boxs_at_frame(self, frame_idx):

        values = []
        for k in range(self.nbr_object):
            obj_traj = self.obj_dict[k]

            idx = next((i for i, v in enumerate(obj_traj["frame"]) if v == frame_idx), None)
            if idx:
                i = idx
                val = [obj_traj["cx"][i], obj_traj["cy"][i], obj_traj["w"][i], obj_traj["h"][i]]
                values.append(val)
            else:
                values.append(None)

        return values

    def masks_at_frame(self, frame_idx, frame):

        boxes = self.boxs_at_frame(frame_idx)

        masks = []
        for box in boxes:

            if box != None:
                mask = box_to_mask(frame, box)
                masks.append(mask)
            else:
                masks.append(None)

        return masks

class wand_trajectory:

    def __init__(self, path):
        
        self.wand_dict =  {"frame" : [], "cx" : [], "cy" : [], "w" : [], "h" : []}

        with open(path, "r") as f:
            next(f)
            for line in f:
                frame, tid, cx, cy, w, h = line.strip().split(",")
                frame, tid = map(int, (frame, tid))
                cx, cy, w, h = map(float, (cx, cy, w, h))

                self.wand_dict["frame"].append(frame)
                self.wand_dict["cx"].append(cx)
                self.wand_dict["cy"].append(cy)
                self.wand_dict["w"].append(w)
                self.wand_dict["h"].append(h)

    def box_at_frame(self, frame_idx):

        idx = next((i for i, v in enumerate(self.wand_dict["frame"]) if v == frame_idx), None)
        if idx is not None:
            i = idx
            val = [self.wand_dict["cx"][i], self.wand_dict["cy"][i], self.wand_dict["w"][i], self.wand_dict["h"][i]]
        else:
            val = None
        
        return val

    def mask_at_frame(self, frame_idx, frame):

        box = self.box_at_frame(frame_idx)
        if box != None:
            mask = box_to_mask(frame, box)
        else:
            mask = None

        return mask
