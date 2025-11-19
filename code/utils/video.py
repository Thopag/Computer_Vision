import cv2
import json

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
        if fixed :
            return data["video_fixed"]["illusions"][trick_id-1]["start_time"]
        else : 
            return data["video_dynamic"]["illusions"][trick_id-1]["start_time"]
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {json_path}")
 
def get_number_of_frames(cap, json_path, fixed):

    fps = cap.get(cv2.CAP_PROP_FPS)

    # in second
    start_trick1 = 0
    tmp= load_json(json_path , fixed , 2)
    start_trick2 =time_to_seconds(tmp)
    end_trick1 = start_trick2
    tmp =load_json(json_path , fixed , 3)
    end_trick2 = time_to_seconds(tmp)
    start_trick3 = end_trick2
    
    # in frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_trick1 = int(start_trick1*fps)
    start_trick2 = int(start_trick2*fps)
    start_trick3 = int(start_trick3*fps)
    end_trick1   = int(end_trick1*fps)
    end_trick2   = int(end_trick2*fps)
    end_trick3   = int(total_frames)

    nbr_trick1 = end_trick1 - start_trick1
    nbr_trick2 = end_trick2 - start_trick2
    nbr_trick3 = end_trick3 - start_trick3

    return nbr_trick1, nbr_trick2, nbr_trick3

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
