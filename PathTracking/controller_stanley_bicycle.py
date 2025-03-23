from PathTracking.controller import Controller
import PathTracking.utils as utils
import sys
import numpy as np
sys.path.append("..")


class ControllerStanleyBicycle(Controller):
    def __init__(self, kp=0.5):
        self.path = None
        self.kp = kp

    # State: [x, y, yaw, delta, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None

        # Extract State
        x, y, yaw, delta, v, l = info["x"], info["y"], info["yaw"], info["delta"], info["v"], info["l"]

        # Search Front Wheel Target
        front_x = x + l*np.cos(np.deg2rad(yaw))
        front_y = y + l*np.sin(np.deg2rad(yaw))
        vf = v / np.cos(np.deg2rad(delta))
        min_idx, min_dist = utils.search_nearest(self.path, (front_x, front_y))
        target = self.path[min_idx]

        # TODO: Stanley Control for Bicycle Kinematic Model
        # Calculate heading error (theta_e)
        target_yaw = target[2]
        heading_error = utils.angle_norm(target_yaw - yaw)

        # Calculate cross-track error with sign
        # Determine if the front wheel is on the left or right side of the path
        path_vector = np.array(
            [np.cos(np.deg2rad(target_yaw)), np.sin(np.deg2rad(target_yaw))])
        error_vector = np.array([front_x - target[0], front_y - target[1]])
        cross_track_error = np.cross(path_vector, error_vector)

        # Apply Stanley control law
        # δ = heading_error + arctan(k * cross_track_error / velocity)
        if vf < 0.1:  # Avoid division by zero
            vf = 0.1
        stanley_term = np.arctan2(self.kp * cross_track_error, vf)
        # Limit steering angle
        next_delta = np.rad2deg(np.clip(heading_error + stanley_term, -30, 30))

        return next_delta, target
