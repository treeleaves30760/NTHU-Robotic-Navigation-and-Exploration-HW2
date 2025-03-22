from PathTracking.controller import Controller
import PathTracking.utils as utils
import sys
import numpy as np
sys.path.append("..")


class ControllerPIDBasic(Controller):
    def __init__(self, kp=0.4, ki=0.0001, kd=0.5):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0

    def set_path(self, path):
        super().set_path(path)
        self.acc_ep = 0
        self.last_ep = 0

    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None

        # Extract State
        x, y, dt = info["x"], info["y"], info["dt"]

        # Search Nesrest Target
        min_idx, min_dist = utils.search_nearest(self.path, (x, y))
        target = self.path[min_idx]

        # TODO: PID Control for Basic Kinematic Model
        # Calculate cross track error
        yaw = info["yaw"]

        # Vector from car to target
        vec_target = np.array([target[0] - x, target[1] - y])
        # Normal vector of car's heading
        vec_normal = np.array(
            [np.cos(np.deg2rad(yaw + 90)), np.sin(np.deg2rad(yaw + 90))])
        # Cross track error (lateral deviation)
        cross_track_error = np.dot(vec_target, vec_normal)

        # PID control
        self.acc_ep += cross_track_error * dt
        de = (cross_track_error - self.last_ep) / dt if dt > 0 else 0
        self.last_ep = cross_track_error

        # Calculate angular velocity based on PID output
        # Positive when need to turn left
        next_w = self.kp * cross_track_error + self.ki * self.acc_ep + self.kd * de

        # Normalize angular velocity
        next_w = utils.angle_norm(next_w)

        return next_w, target
