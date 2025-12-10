out_path = "output/trick_2_state.mp4"
def func(cap: cv2.VideoCapture , writer : cv2.VideoWriter) -> int: 
    state = False
    nb_frames = 0  
    switch = 0
    d1 =0
    d0 =0
    count = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        # Process frame_bgr here
        #===============================
        # Detect blue and red colors (assuming good enough)
        mask_blue = detect_color(frame_bgr, [0,0,255] ,[100,100],[255,255], tuning=25)
        mask_red = detect_color(frame_bgr, [255,0,5] ,[55,55],[255,255], tuning=25)
        _,mask_blue = roi(mask_blue, 10, 10)
        _,mask_red =  roi(mask_red, 3, 10)
        #output  = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_blue)
        #output += cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_red)
        #===============================
        # Start the trick 
        _ ,roi_blue = roi(mask_blue, 10, 20)
        _ ,roi_red  = roi(mask_red, 3, 20)
        overlap = cv2.bitwise_and(roi_blue, roi_red)
        
        
        # Count number of overlapping pixels
        overlap_pixels = np.count_nonzero(overlap)
        # Mark as overlapping
        string = f"Overlap pixels: {overlap_pixels}"
        cv2.putText(frame_bgr, string,  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        
        #===============================

        if overlap_pixels > 30:
            nb_frames += 1
            if nb_frames >= 3:
                state = True
                d1 += 1
                string += " - TRICK DETECTED!"
                nb_frames = 0  
            else :
                switch +=1
        else :
            nb_frames = 0
        string = f"Frame: {nb_frames}"

        
        cv2.putText(frame_bgr, string, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        string = f"State: {state}"
        cv2.putText(frame_bgr, string, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        string = f"Switches: {switch}"
        cv2.putText(frame_bgr, string, (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        string = f"d1: {d1}"
        cv2.putText(frame_bgr, string, (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        writer.write(frame_bgr)
    return 1 
    