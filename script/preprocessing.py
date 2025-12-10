import time

from script.CONFIG import FILE_NAME
from script.utils.tracking.interpolation import interpolation_object, interpolation_wand
from script.utils.tracking.visual_trajectory import visualizer_object, visualizer_wand
from script.utils.tracking.object_tracker import object_tracking
from script.utils.tracking.wand_tracker import wand_tracking

if __name__ == "__main__":

    start_time = time.time()

    obj_tracking_video = f"files/object_tracking/{FILE_NAME}.mp4"
    obj_tracking_log = f"files/object_tracking/{FILE_NAME}.txt"

    interpolation_obj_txt = f"files/interpolation/object_{FILE_NAME}.txt"
    interpolation_obj_video = f"files/interpolation/visual_object_{FILE_NAME}.mp4"

    wand_tracking_video = f"files/wand_tracking/{FILE_NAME}.mp4"
    wand_tracking_log =  f"files/wand_tracking/{FILE_NAME}.txt"

    interpolation_wand_txt = f"files/interpolation/wand_{FILE_NAME}.txt"
    interpolation_wand_video = f"files/interpolation/visual_wand_{FILE_NAME}.mp4"

    print("------ Start object tracking ------")
    t = time.time()
    object_tracking(obj_tracking_video, obj_tracking_log)
    obj_track_time = time.time() - t
    print("------ End object tracking ------")

    print("------ Start object interpolation ------")
    t = time.time()
    interpolation_object(obj_tracking_log, interpolation_obj_txt)
    inter_obj_time = time.time() - t
    print("------ End object interpolation ------")

    print("------ Start visual object ------")
    t = time.time()
    visualizer_object(interpolation_obj_txt, interpolation_obj_video)
    visual_obj_time = time.time() - t
    print("------ End visual object ------")

    print("------ Start wand tracking ------")
    t = time.time()
    wand_tracking(obj_tracking_log, wand_tracking_video, wand_tracking_log)
    wank_track_time = time.time() - t
    print("------ End wand tracking ------")

    print("------ Start interpolation wand ------")
    t = time.time()
    interpolation_wand(wand_tracking_log, interpolation_wand_txt)
    inter_wand_time = time.time() - t
    print("------ End interpolation wand ------")

    print("------ Start visual wand ------")
    t = time.time()
    visualizer_wand(interpolation_wand_txt, interpolation_wand_video)
    visual_wand_time = time.time() - t
    print("------ End visual wand ------")

    end_time = time.time()

    print("=============================================")
    print(f"Object tracking : {obj_track_time:.4f} seconds")
    print(f"Object interpolation : {inter_obj_time:.4f} seconds")
    print(f"Visual object : {visual_obj_time:.4f} seconds")
    print(f"Wand tracking : {wank_track_time:.4f} seconds")
    print(f"Interpolation wand : {inter_wand_time:.4f} seconds")
    print(f"Visual wand : {visual_wand_time:.4f} seconds")
    print(f"The total execution time is :{end_time-start_time:.4f}")
    print("=============================================")