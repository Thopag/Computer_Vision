import json
def get_trick_start_frames(json_path, fps, video_type="video_fixed"):
    """
    Reads the JSON annotation file and returns start frame indices for all three tricks.

    Args:
        json_path (str): Path to the JSON file
        fps (float): Frames per second of the video (should be given or extracted using OpenCV)
        video_type (str): 'video_fixed' or 'video_dynamic'

    Returns:
        dict: {trick_id: start_frame_index}
    """
    def time_to_seconds(time_str):
        h, m, s = map(int, time_str.split(':'))
        return h*3600 + m*60 + s

    with open(json_path, 'r') as f:
        data = json.load(f)

    illusions = data[video_type]["illusions"]
    start_frames = {}

    for illusion in illusions:
        trick_id = illusion["trick_id"]
        start_time_sec = time_to_seconds(illusion["start_time"])
        start_frames[trick_id] = int(start_time_sec * fps)

    return start_frames


