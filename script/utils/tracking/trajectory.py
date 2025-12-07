import numpy as np
from ..box import box_to_mask

class object_trajectory:

    def __init__(self, path, nbr_object):
        
        self.nbr_object = nbr_object
        self.obj_dict =  {int(i): {"frame" : [], "cx" : [], "cy" : [], "w" : [], "h" : []} for i in range(nbr_object)}

        with open(path, "r") as f:
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

        values = self.boxs_at_frame(frame_idx)

        masks = []
        for val in values:

            if val != None:
                mask = box_to_mask(frame, val[0], val[1], val[2], val[3])
                masks.append(mask)
            else:
                masks.append(None)

        return masks
