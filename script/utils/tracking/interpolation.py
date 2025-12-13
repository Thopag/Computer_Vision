import numpy as np
from scipy.interpolate import interp1d

from .trajectory import object_trajectory, wand_trajectory
from ...CONFIG import *

def make_interpolation(obj_dict):

    frames = np.array(obj_dict["frame"], dtype=float)
    cx = np.array(obj_dict["cx"], dtype=float)
    cy = np.array(obj_dict["cy"], dtype=float)
    h = np.array(obj_dict["h"], dtype=float)
    w = np.array(obj_dict["w"], dtype=float)

    if len(frames) == 0:
        print("NO TRAJECTORY")
        return None

    f_x = interp1d(frames, cx, kind=INTERPOLATION_TYPE)
    f_y = interp1d(frames, cy, kind=INTERPOLATION_TYPE)
    f_h = interp1d(frames, h, kind=INTERPOLATION_TYPE)
    f_w = interp1d(frames, w, kind=INTERPOLATION_TYPE)

    new_frames = np.arange(frames[0], frames[-1]+1, 1, dtype=float)
    new_cx = f_x(new_frames)
    new_cy = f_y(new_frames)
    new_w = f_w(new_frames)
    new_h = f_h(new_frames)

    new_obj_dict =  {"frame" : new_frames.astype(int), "cx" : new_cx.astype(int), "cy" : new_cy.astype(int), "w" : new_w.astype(int), "h" : new_h.astype(int)} 
    return new_obj_dict

def write_trajectory_txt(obj_dict, id, f):

    if obj_dict == None:
        return

    for i in range(len(obj_dict["frame"])):

        tid = id
        frame_idx = obj_dict["frame"][i]
        cx = obj_dict["cx"][i]
        cy = obj_dict["cy"][i]
        w = obj_dict["w"][i]
        h = obj_dict["h"][i]

        f.write(f"{frame_idx},{tid},{cx},{cy},{w},{h}\n")

    return

def interpolation_object(object_path, out_txt):

    f = open(out_txt, "w")
    
    obj_trajs = object_trajectory(object_path)

    f.write(f"nbr_object : {obj_trajs.nbr_object}\n")
    f.write("Frame,ID,cx,cy,w,h\n")

    for i in range(obj_trajs.nbr_object):
        obj_dict = make_interpolation(obj_trajs.obj_dict[i])
        write_trajectory_txt(obj_dict, str(i), f)

    return

def interpolation_wand(wand_path, out_txt):

    f = open(out_txt, "w")
    f.write("Frame,ID,cx,cy,w,h\n")

    wand_traj = wand_trajectory(wand_path)

    wand_traj = make_interpolation(wand_traj.wand_dict)
    write_trajectory_txt(wand_traj, "-1", f)

    return 

if __name__ == "__main__":

    obj_tracking_log = f"files/object_tracking/{FILE_NAME}_{N_OBJECT}_obj.txt"
    interpolation_obj_txt = f"files/interpolation/{FILE_NAME}_{N_OBJECT}_obj.txt"

    interpolation_object(obj_tracking_log, interpolation_obj_txt)

