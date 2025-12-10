import cv2
def draw_bounding_box_of_object_detected(cap: cv2.VideoCapture, writer: cv2.VideoWriter):

    switch = 0
    overlap_prev = False
    frame_idx = 0

    last_masks = None
    last_labels = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        output = frame.copy()

        # Run object detection every 20 frames
        if frame_idx % 20 == 0:
            last_masks, last_labels = detect_objects(frame)

        # -------------------------------
        # Draw bounding boxes for each object
        # -------------------------------
        if last_masks is not None:
            for mask, label in zip(last_masks, last_labels):

                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()

                    cv2.rectangle(output,
                                  (x_min, y_min),
                                  (x_max, y_max),
                                  (0, 255, 0),
                                  2)
                    cv2.putText(output, str(label),
                                (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

        # -------------------------------
        # (Optional) Your wand logic here
        # -------------------------------
        # Example:
        # cv2.putText(output, f"Switch: {switch}",
        #             (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        writer.write(output)
        frame_idx += 1

    return 1