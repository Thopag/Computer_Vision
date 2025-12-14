import numpy as np
from scipy.interpolate import interp1d

from .trajectory import object_trajectory, wand_trajectory
from script.CONFIG import *

def make_interpolation(traj_dict):
    """
    Take a trajectory dict and return the interpolated version of the dict.

    Args:
        traj_dict: The traj_dict to interpolate

    Return:
        The interpolated traj_dict
    """

    # Get all the values that need to be interpolated as a function of the frame
    frames = np.array(traj_dict["frame"], dtype=float)
    cx = np.array(traj_dict["cx"], dtype=float)
    cy = np.array(traj_dict["cy"], dtype=float)
    h = np.array(traj_dict["h"], dtype=float)
    w = np.array(traj_dict["w"], dtype=float)

    # Check if not empty
    if len(frames) == 0:
        print("NO TRAJECTORY")
        return None

    # Make interpolation functions
    f_x = interp1d(frames, cx, kind=INTERPOLATION_TYPE)
    f_y = interp1d(frames, cy, kind=INTERPOLATION_TYPE)
    f_h = interp1d(frames, h, kind=INTERPOLATION_TYPE)
    f_w = interp1d(frames, w, kind=INTERPOLATION_TYPE)

    # Create interpolated vectors
    new_frames = np.arange(frames[0], frames[-1]+1, 1, dtype=float)
    new_cx = f_x(new_frames)
    new_cy = f_y(new_frames)
    new_w = f_w(new_frames)
    new_h = f_h(new_frames)

    new_traj_dict =  {"frame" : new_frames.astype(int), "cx" : new_cx.astype(int), "cy" : new_cy.astype(int), "w" : new_w.astype(int), "h" : new_h.astype(int)} 
    return new_traj_dict

def write_trajectory_txt(traj_dict, id, f):
    """
    Create the trajectory of the given traj dict in the file f .txt
    
    Args:
        traj_dict: The dict countaining the values to write
        id: The id given to the trajectory (obj or wand id)
        f: The file.txt where the value are stored
    """

    if traj_dict == None:
        return

    for i in range(len(traj_dict["frame"])):

        tid = id
        frame_idx = traj_dict["frame"][i]
        cx = traj_dict["cx"][i]
        cy = traj_dict["cy"][i]
        w = traj_dict["w"][i]
        h = traj_dict["h"][i]

        f.write(f"{frame_idx},{tid},{cx},{cy},{w},{h}\n")

    return

def interpolation_object(object_path, out_txt):
    """
    Given the object trajectory, re write a interpolate version

    Args:
        object_path: The file.txt containing the trajectories to interpolate.
        out_txt: The output .txt
    """

    f = open(out_txt, "w")
    
    obj_trajs = object_trajectory(object_path)

    f.write(f"nbr_object : {obj_trajs.nbr_object}\n")
    f.write("Frame,ID,cx,cy,w,h\n")

    for i in range(obj_trajs.nbr_object):
        obj_dict = make_interpolation(obj_trajs.obj_dict[i])
        write_trajectory_txt(obj_dict, str(i), f)

    return

def interpolation_wand(wand_path, out_txt):
    """
    Given the wand trajectory, re write a interpolate version

    Args:
        wand_path: The file.txt containing the trajectory to interpolate.
        out_txt: The output .txt
    """

    f = open(out_txt, "w")
    f.write("Frame,ID,cx,cy,w,h\n")

    wand_traj = wand_trajectory(wand_path)

    wand_traj = make_interpolation(wand_traj.wand_dict)
    write_trajectory_txt(wand_traj, "-1", f)

    return 

if __name__ == "__main__":

    obj_tracking_log = f"files/object_tracking/{FILE_NAME}_{N_OBJECT}_obj.txt"
    interpolation_obj_txt = f"files/interpolation/{FILE_NAME}_{N_OBJECT}_obj.txt"

    #interpolation_object(obj_tracking_log, interpolation_obj_txt)

    wand_2_tracking_log =  f"files/wand_tracking/{FILE_NAME}_trick2.txt"
    interpolation_wand_2_txt = f"files/interpolation/{FILE_NAME}_wand_trick2.txt"

    #interpolation_object(wand_2_tracking_log, interpolation_wand_2_txt)

    # special #

    obj_backup_tracking_log = f"files/object_tracking/{FILE_NAME}_{N_OBJECT}_obj_backup.txt"
    interpolation_obj_backup_txt = f"files/interpolation/{FILE_NAME}_{N_OBJECT}_obj_backup.txt"

    #interpolation_object(obj_backup_tracking_log, interpolation_obj_backup_txt)

    obj_static_tracking_log = f"files/object_tracking/{FILE_NAME}_{N_OBJECT}_obj_static.txt"
    interpolation_static_txt = f"files/interpolation/{FILE_NAME}_{N_OBJECT}_obj_static.txt"

    #interpolation_object(obj_static_tracking_log, interpolation_static_txt)