class WandMotion:
    def __init__(self):
        self.prev = None

    def compute(self, pos):
        if pos is None:
            return 0, 0

        if self.prev is None:
            self.prev = pos
            return 0, 0

        dx = pos[0] - self.prev[0]
        dy = pos[1] - self.prev[1]

        self.prev = pos
        return dx, dy
