import time

from script.CONFIG import *
from script.utils.tracking.interpolation import interpolation_object, interpolation_wand
from script.utils.tracking.visual_trajectory import visualizer_object, visualizer_wand, visualizer_combined
from script.utils.tracking.object_tracker import object_tracking
from script.utils.tracking.wand_tracker import wand_tracking

# --- 3 object --- #

obj_tracking_video = f"files/object_tracking/{FILE_NAME}_{N_OBJECT}_obj.mp4"
obj_tracking_log = f"files/object_tracking/{FILE_NAME}_{N_OBJECT}_obj.txt"

interpolation_obj_txt = f"files/interpolation/{FILE_NAME}_{N_OBJECT}_obj.txt"
interpolation_obj_video = f"files/interpolation/{FILE_NAME}_visual_{N_OBJECT}_obj.mp4"

# --- wand trick 2 --- #

wand_2_tracking_video = f"files/wand_tracking/{FILE_NAME}_trick2.mp4"
wand_2_tracking_log =  f"files/wand_tracking/{FILE_NAME}_trick2.txt"

interpolation_wand_2_txt = f"files/interpolation/{FILE_NAME}_wand_trick2.txt"
interpolation_wand_2_video = f"files/interpolation/{FILE_NAME}_visual_wand_trick2.mp4"

# --- ball only --- #

ball_tracking_video = f"files/object_tracking/{FILE_NAME}_ball.mp4"
ball_tracking_log = f"files/object_tracking/{FILE_NAME}_ball.txt"

interpolation_ball_txt = f"files/interpolation/{FILE_NAME}_ball.txt"
interpolation_ball_video = f"files/interpolation/{FILE_NAME}_visual_ball.mp4"

# --- wand trick3 --- #

wand_3_tracking_video = f"files/wand_tracking/{FILE_NAME}_trick3.mp4"
wand_3_tracking_log =  f"files/wand_tracking/{FILE_NAME}_trick3.txt"

interpolation_wand_3_txt = f"files/interpolation/{FILE_NAME}_wand_trick3.txt"
interpolation_wand_3_video = f"files/interpolation/{FILE_NAME}_visual_wand_trick3.mp4"

# --- interation for trick 2 --- #

interaction_txt = f"files/object_&_wand/{FILE_NAME}_interaction_object_&_wand.txt"
interaction_video = f"files/object_&_wand/{FILE_NAME}_interaction_object_&_wand.mp4"
output_all_objects = f"files/object_&_wand/{FILE_NAME}_all_object_&_wand.txt"

if __name__ == "__main__":

    start_time = time.time()

    # --------------------------- Objects for trick 1 & 2 --------------------------- #

    # print("------ Start object tracking ------")
    # t = time.time()
    # object_tracking(obj_tracking_video, obj_tracking_log, 
    #                     N_OBJECT, OBJ_TRACKER_START_FRAME, end_frame=OBJ_TRACKER_END_FRAME)
    # obj_track_time = time.time() - t
    # print("------ End object tracking ------")

    # print("------ Start object interpolation ------")
    # t = time.time()
    # interpolation_object(obj_tracking_log, interpolation_obj_txt)
    # inter_obj_time = time.time() - t
    # print("------ End object interpolation ------")

    # print("------ Start visual object ------")
    # t = time.time()
    # visualizer_object(interpolation_obj_txt, interpolation_obj_video)
    # visual_obj_time = time.time() - t
    # print("------ End visual object ------")

    # # --------------------------- wand for trick 2 --------------------------- #

    print("------ Start wand tracking ------")
    t = time.time()
    wand_tracking(obj_tracking_log, wand_2_tracking_video, wand_2_tracking_log, 
                    start_frame=WAND_2_TRACKER_START_FRAME, end_frame=WAND_2_TRACKER_END_FRAME)
    wank_2_track_time = time.time() - t
    print("------ End wand tracking ------")

    print("------ Start interpolation wand ------")
    t = time.time()
    interpolation_wand(wand_2_tracking_log, interpolation_wand_2_txt)
    inter_wand_2_time = time.time() - t
    print("------ End interpolation wand ------")

    print("------ Start visual wand ------")
    t = time.time()
    visualizer_wand(interpolation_wand_2_txt, interpolation_wand_2_video)
    visual_wand_2_time = time.time() - t
    print("------ End visual wand ------")

    # # --------------------------- Objects for trick 3 (only ball) --------------------------- #

    # print("------ Start object tracking ------")
    # t = time.time()
    # object_tracking(ball_tracking_video, ball_tracking_log, 
    #                     1, OBJ_BALL_TRACKER_START_FRAME, end_frame=OBJ_BALL_TRACKER_END_FRAME)
    # ball_track_time = time.time() - t
    # print("------ End object tracking ------")

    # print("------ Start object interpolation ------")
    # t = time.time()
    # interpolation_object(ball_tracking_log, interpolation_ball_txt)
    # inter_ball_time = time.time() - t
    # print("------ End object interpolation ------")

    # print("------ Start visual object ------")
    # t = time.time()
    # visualizer_object(interpolation_ball_txt, interpolation_ball_video)
    # visual_ball_time = time.time() - t
    # print("------ End visual object ------")

    #--------------------------- wand for trick 3 --------------------------- #

    # print("------ Start wand tracking ------")
    # t = time.time()
    # wand_tracking(obj_tracking_log, wand_3_tracking_video, wand_3_tracking_log, 
    #                 start_frame = WAND_3_TRACKER_START_FRAME, end_frame=WAND_3_TRACKER_END_FRAME)
    # wank_3_track_time = time.time() - t
    # print("------ End wand tracking ------")

    # print("------ Start interpolation wand ------")
    # t = time.time()
    # interpolation_wand(wand_3_tracking_log, interpolation_wand_3_txt)
    # inter_wand_3_time = time.time() - t
    # print("------ End interpolation wand ------")

    # print("------ Start visual wand ------")
    # t = time.time()
    # visualizer_wand(interpolation_wand_3_txt, interpolation_wand_3_video)
    # visual_wand_3_time = time.time() - t
    # print("------ End visual wand ------")

    # print("------ Start visual interaction ------")
    # t = time.time()
    # visualizer_combined(interpolation_wand_2_txt, interpolation_obj_txt, 
    #                         interaction_video, output_all_objects, interaction_txt)
    # interation_time = time.time() - t
    # print("------ End visual interaction ------")


    end_time = time.time()

    print("=============================================")
    # print(f"Object tracking : {obj_track_time:.4f} seconds")
    # print(f"Object interpolation : {inter_obj_time:.4f} seconds")
    # print(f"Visual object : {visual_obj_time:.4f} seconds")
    # print(f"Wand 2 tracking : {wank_2_track_time:.4f} seconds")
    # print(f"Interpolation wand 2 : {inter_wand_2_time:.4f} seconds")
    # print(f"Visual wand 2 : {visual_wand_2_time:.4f} seconds")

    # print(f"Ball tracking : {ball_track_time:.4f} seconds")
    # # print(f"Ball interpolation : {inter_ball_time:.4f} seconds")
    # # print(f"Visual ball : {visual_ball_time:.4f} seconds")
    # print(f"Wand 3 tracking : {wank_3_track_time:.4f} seconds")
    # print(f"Interpolation wand 3 : {inter_wand_3_time:.4f} seconds")
    # # print(f"Visual wand 3 : {visual_wand_3_time:.4f} seconds")

    # print(f"Visual interation : {interation_time:.4f} seconds")
    # print(f"The total execution time is :{end_time-start_time:.4f}")
    print("=============================================")