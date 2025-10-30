import cv2
import numpy as np
import matplotlib.pyplot as plt
import json 

def time_to_seconds(time_str):
    """Convert 'HH:MM:SS' to seconds."""
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s

def time_to_hms(seconds):
    """Convert seconds to 'HH:MM:SS'."""
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = int(round(seconds % 60))
    return f"{h:02}:{m:02}:{s:02}"

def load_json(json_path , fixed , trick_id):
    
    """load json start time based on the trick_id
       fixed  : a boolean  , TRUE = FIXED VIDEO , FALSE = DYNAMIC
       trick_id :  1,2,3 

    Raises:
        FileNotFoundError

    Returns:
        (string): start time of the needed trick in HH:MM:SS format
    """

    try :
        with open(json_path, "r") as f:
            data = json.load(f)
        print("JSON file loaded successfully.")
        if fixed :
            return data["video_fixed"]["illusions"][trick_id-1]["start_time"]
        else : 
            return data["video_dynamic"]["illusions"][trick_id-1]["start_time"]
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {json_path}")
 

def apply_function_to_video(in_path, out_path, func):
    """
    take an input path  an output path and a function to apply to each frame
    source : code from TP1 

    Args:
        func :a function that applies to a frame and returns the processed frame

    Raises:
        IOError: 
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise IOError(f"Could not open {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        
        output = func(frame_bgr)
        writer.write(output)

    cap.release()
    writer.release()
    print("Saved:", out_path)

def process(path, func1,func2,func3):
    """
    take a video as input and apply the corresponding function
    to each trick based on the annotations in the json file.
    for example func1 to trick 1
    etc.
    source : code from TP1 

    Args:
        func_i :a function that applies to a frame and returns the processed frame
        path (dict) : dictionary that contains the input video path, output video path and json path
    Raises:
        IOError: 
    """
    
    in_path  = path["in_path"]
    out_path = path["out_path"]
    json_path= path["json_path"]
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise IOError(f"Could not open {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    t1 = load_json(json_path , True , 1)
    t2 = load_json(json_path , True , 2)
    t3 = load_json(json_path , True , 3)
    print("trick 1 start",t1)
    print("trick 2 start",t2)
    print("trick 3 start",t3)
    start_sec1 = time_to_seconds(t1)
    start_sec2 = time_to_seconds(t2)
    start_sec3 = time_to_seconds(t3)
    
    start_frame1 = int(start_sec1 * fps)
    start_frame2 = int(start_sec2 * fps)
    start_frame3 = int(start_sec3 * fps)
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        # trick 1 did not start yet so no processing
        if frame_idx <start_frame1:
            output = frame_bgr
        # trick 1 started 
        elif frame_idx>=start_frame1 and frame_idx <start_frame2:
            output = func1(frame_bgr)
        #trick 2 started    
        elif frame_idx>=start_frame2 and frame_idx <start_frame3:
            output = func2(frame_bgr)
        #trick 3 started
        else:   
            output = func3(frame_bgr)
        
        frame_idx += 1
        writer.write(output)

    cap.release()
    writer.release()
    print("Saved:", out_path)
    print("no processing from ","00:00:00","to", time_to_hms(start_frame1/fps))
    print("trick 1 processing from",time_to_hms(start_frame1/fps), "to", time_to_hms(start_frame2/fps))
    print("trick 2 processing from  ",time_to_hms(start_frame2/fps),"to", time_to_hms(start_frame3/fps))
    print("trick 3 processing from " ,time_to_hms(start_frame3/fps) ,"to end")