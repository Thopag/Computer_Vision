class BallMotion:
    # To modify to change sensibility of clone ball deplacement
    def __init__(self, sensitivity_x=2.0, sensitivity_y=0.25):
        self.sx = sensitivity_x
        self.sy = sensitivity_y

    def apply(self, ball_pos, wand_dx, wand_dy):
        bx, by = ball_pos
        bx += int(self.sx * wand_dx)
        by += int(self.sy * wand_dy)
        return bx, by
