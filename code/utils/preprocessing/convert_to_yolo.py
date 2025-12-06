from utils.yolo import annotate_video

# for i in [1, 3, 4, 5, 6, 8, 9, 10, 11, 12]:

#     print(f"-----------------{i} fixed-------------------")
#     in_path  = f"../input/video_group_{i}_fixed.mp4"
#     annotate_video(in_path, read_every_x_frame=5, out_path=None, confidence_threshold=0.0)

for i in [1, 3, 4, 5, 6, 8, 9, 10, 11, 12]:

    print(f"-----------------{i} dynamic-------------------")
    in_path  = f"../input/video_group_{i}_dynamic.mp4"
    annotate_video(in_path, read_every_x_frame=5, out_path=None, confidence_threshold=0.0)