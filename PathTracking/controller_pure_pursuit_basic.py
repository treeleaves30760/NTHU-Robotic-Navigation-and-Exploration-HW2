from PathTracking.controller import Controller
import PathTracking.utils as utils
import sys
import numpy as np
sys.path.append("..")


class ControllerPurePursuitBasic(Controller):
    def __init__(self, kp=1, Lfc=10):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc

    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None

        # Extract State
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]

        # Search Front Target
        min_idx, min_dist = utils.search_nearest(self.path, (x, y))
        Ld = self.kp*v + self.Lfc
        target_idx = min_idx
        for i in range(min_idx, len(self.path)-1):
            dist = np.sqrt((self.path[i+1, 0]-x)**2 + (self.path[i+1, 1]-y)**2)
            if dist > Ld:
                target_idx = i
                break
        target = self.path[target_idx]

        # TODO: Pure Pursuit Control for Basic Kinematic Model
        # Calculate the angle between vehicle heading and target point
        alpha = np.arctan2(target[1] - y, target[0] - x) - np.deg2rad(yaw)

        # Calculate angular velocity based on look-ahead distance
        # For basic model, we directly set angular velocity
        if v > 0.1:  # Avoid division by zero or very small values
            next_w = np.rad2deg(2 * v * np.sin(alpha) / Ld)
        else:
            # If velocity is very small, just point towards the target
            # Multiply by a factor for faster convergence
            next_w = np.rad2deg(alpha) * 2

        # Normalize the angular velocity
        next_w = utils.angle_norm(next_w)

        return next_w, target
