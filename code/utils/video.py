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