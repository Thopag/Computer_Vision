from .utils.yolo_detector import yolo_video

in_path  = f"input/video_group_1_dynamic.mp4"

blacklist = ['person', 'chair', 'dining table', 'train', 'bus', 'tennis racket', 'frisbee'
    , 'handbag', 'cake', 'tv', 'orange', 'book', 'cup', 'skateboard', 'laptop', 'refrigerator', 'remote', 'backpack', 'bowl']

blacklist = []

yolo_video(in_path, read_every_x_frame=5, blacklist=blacklist, confidence_threshold=0.0)