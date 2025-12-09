class MagicianDetector:
    """
    Extracts person bounding box from YOLO detections.
    """

    def detect(self, detections):
        """
        detections = list from YOLODetector.detect()
        returns (center_x, center_y), (x1,y1,x2,y2)
        """

        magician_box = None
        magician_center = None

        for cls, x1, y1, x2, y2, conf in detections:
            if cls.lower() == "person":
                magician_box = (x1, y1, x2, y2)
                magician_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        return magician_center, magician_box
