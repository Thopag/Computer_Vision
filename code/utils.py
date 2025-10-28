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
 
def detect_color(image, color ,lower,upper, tuning=25):
    """
    Detect a specific color in an image and create a mask.

    Args:
        image (np.array) : and input image in BGR format
        color (tuple): a list RGB color like [255,0,0]
        lower,upper (tuple) :  lower and upper value of s & v in hsv format
        tuning (int, optional):tuning parameter. Defaults to 25.

    Returns:
        (np.array): mask corresponding to the detected color
        
    """
    lower_s = lower[0]
    lower_v = lower[1]
    upper_s = upper[0]
    upper_v = upper[1]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color = np.uint8([[color]])
    
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_color)
    H = h[0][0]
    print("Hue value of the target color:", H)
    lower_h = max(H - tuning, 0)
    upper_h = min(H + tuning, 179)
    lower = np.array([lower_h, lower_s, lower_v])
    upper = np.array([upper_h, upper_s, upper_v])
    
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def change_color_mask(image, orig_rgb, target_rgb, mask):
    """
    Change a specific color in an image to a target color based on the mask given as input.

    Args:
        image (np.array):  and input image in BGR format 
        orig_rgb (tuple): origin color in RGB format like [255,0,0]
        target_rgb (tuple): target color in RGB format like [0,255,0]
        mask (np.array) :  The mask to be used to change the color
    Returns:
        (np.array): image with the color changed in BGR format
        """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    orig_hsv = cv2.cvtColor(np.uint8([[orig_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    target_hsv = cv2.cvtColor(np.uint8([[target_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    delta_h = (int(target_hsv[0]) - int(orig_hsv[0])) % 180
    if delta_h > 90:
        delta_h -= 180

    delta_s = int(target_hsv[1]) - int(orig_hsv[1])
    delta_v = int(target_hsv[2]) - int(orig_hsv[2])
    
    h, s, v = cv2.split(hsv)
    h = np.mod(h.astype(np.int16) + delta_h, 180).astype(np.uint8)
    s = np.clip(s.astype(np.int16) + delta_s, 0, 255).astype(np.uint8)
    v = np.clip(v.astype(np.int16) + delta_v, 0, 255).astype(np.uint8)

    
    hsv_shifted = cv2.merge([h, s, v])
    bgr_shifted = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)
    
    result = cv2.bitwise_or(
        cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask)),
        cv2.bitwise_and(bgr_shifted, bgr_shifted, mask=mask)
    )
    print ("the original color is :" , orig_rgb)
    print ("the target  color is :" , target_rgb)
    print("the color change worked successfully ")
    
    return result

def change_color(image, orig_rgb, target_rgb, lower, upper, tuning=25):
    """
    Change a specific color in an image to a target color. (OLD FUNCTION)

    Args:
        image (np.array):  and input image in BGR format 
        orig_rgb (tuple): origin color in RGB format like [255,0,0]
        target_rgb (tuple): target color in RGB format like [0,255,0]
        lower,upper (tuple) :  lower and upper value of s & v in hsv format
        tuning (int, optional): parameter that allow modifying the hue value, Defaults to 25.

    Returns:
        (np.array): image with the color changed in BGR format
        """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = detect_color(image, orig_rgb, lower,upper, tuning)
    
    orig_hsv = cv2.cvtColor(np.uint8([[orig_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    target_hsv = cv2.cvtColor(np.uint8([[target_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    delta_h = (int(target_hsv[0]) - int(orig_hsv[0])) % 180
    if delta_h > 90:
        delta_h -= 180

    delta_s = int(target_hsv[1]) - int(orig_hsv[1])
    delta_v = int(target_hsv[2]) - int(orig_hsv[2])
    
    h, s, v = cv2.split(hsv)
    h = np.mod(h.astype(np.int16) + delta_h, 180).astype(np.uint8)
    s = np.clip(s.astype(np.int16) + delta_s, 0, 255).astype(np.uint8)
    v = np.clip(v.astype(np.int16) + delta_v, 0, 255).astype(np.uint8)

    
    hsv_shifted = cv2.merge([h, s, v])
    bgr_shifted = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)
    
    result = cv2.bitwise_or(
        cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask)),
        cv2.bitwise_and(bgr_shifted, bgr_shifted, mask=mask)
    )
    print("the color change worked successfully ")
    
    return result

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
    
def roi(img  , RGB ,lower,upper,s_erod , s_dil):
    """
    # could add directly the mask as input
    Create a region of interest mask based
    
    s_erod: size for erosion
    s_dil: size for dilation
    img : input image
    RGB : color to detect
    
    return 
        image of BGR format 
    """
    
    mask = detect_color(img,RGB,lower,upper,tuning = 25)
    # we erode first to remove noise (small white regions detected by mistake)
    kernel_erod = (s_erod , s_erod)
    SE= cv2.getStructuringElement(cv2.MORPH_RECT,kernel_erod)
    eroded_mask = cv2.erode(mask,SE)
    
    #then we dilate to get the full region of interest around the detected object
    
    kernel_dil = (s_dil , s_dil)
    SE= cv2.getStructuringElement(cv2.MORPH_RECT,kernel_dil)
    
    roi_mask = cv2.dilate(eroded_mask,SE)
    
    new_image = cv2.bitwise_and(img , img , mask = roi_mask)
    
    #for visualization
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    return [new_image, roi_mask]
