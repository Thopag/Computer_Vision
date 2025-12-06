import numpy as np
from scipy.interpolate import interp1d
from trajectory import object_trajectory

degree = 'linear'

def make_interpolation(obj_dict):

    frames = np.array(obj_dict["frame"], dtype=float)
    cx = np.array(obj_dict["cx"], dtype=float)
    cy = np.array(obj_dict["cy"], dtype=float)
    h = np.array(obj_dict["h"], dtype=float)
    w = np.array(obj_dict["w"], dtype=float)

    f_x = interp1d(frames, cx, kind=degree)
    f_y = interp1d(frames, cy, kind=degree)
    f_h = interp1d(frames, h, kind=degree)
    f_w = interp1d(frames, w, kind=degree)

    new_frames = np.arange(frames[0], frames[-1]+1, 1, dtype=float)
    new_cx = f_x(new_frames)
    new_cy = f_y(new_frames)
    new_w = f_w(new_frames)
    new_h = f_h(new_frames)

    new_obj_dict =  {"frame" : new_frames.astype(int), "cx" : new_cx.astype(int), "cy" : new_cy.astype(int), "w" : new_w.astype(int), "h" : new_h.astype(int)} 
    return new_obj_dict

def write_trajectory_txt(obj_dict, id, f):

    for i in range(len(obj_dict["frame"])):

        tid = id
        frame_idx = obj_dict["frame"][i]
        cx = obj_dict["cx"][i]
        cy = obj_dict["cy"][i]
        w = obj_dict["w"][i]
        h = obj_dict["h"][i]

        f.write(f"{frame_idx},{tid},{cx},{cy},{w},{h}\n")

    return

if __name__ == "__main__":

    output_txt = "../../output/trackerV2_test.txt"

    obj_trajs = object_trajectory(output_txt, 3)

    obj_dict_1 = make_interpolation(obj_trajs.obj_dict[0])
    obj_dict_2 = make_interpolation(obj_trajs.obj_dict[1])
    obj_dict_3 = make_interpolation(obj_trajs.obj_dict[2])

    new_output_txt = "../../output/test_inter.txt"
    f = open(new_output_txt, "w")
    f.write("Frame,ID,cx,cy,w,h\n")

    write_trajectory_txt(obj_dict_1, 0, f)
    write_trajectory_txt(obj_dict_2, 1, f)
    write_trajectory_txt(obj_dict_3, 2, f)
