import numpy as np

class MagicianMotion:
    """
    Track magician displacement frame-by-frame.
    Used to decide when ball motion is allowed.
    """

    def __init__(self):
        self.prev = None

    def compute(self, pos):
        """
        pos = (cx, cy) of magician (YOLO person center)
        Returns (dx, dy).
        """
        if pos is None:
            return 0, 0

        if self.prev is None:
            self.prev = pos
            return 0, 0

        dx = pos[0] - self.prev[0]
        dy = pos[1] - self.prev[1]

        self.prev = pos
        return dx, dy
