from PathTracking.controller import Controller
import PathTracking.utils as utils
import sys
import numpy as np
sys.path.append("..")


class ControllerPIDBicycle(Controller):
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

        # TODO: PID Control for Bicycle Kinematic Model
        # Calculate cross track error
        yaw = info["yaw"]
        current_x, current_y = x, y
        target_x, target_y = target[0], target[1]

        # Convert to front axle position if we need to
        if "l" in info:
            front_x = current_x + info["l"] * np.cos(np.deg2rad(yaw))
            front_y = current_y + info["l"] * np.sin(np.deg2rad(yaw))
            current_x, current_y = front_x, front_y

        # Vector from car to target
        vec_target = np.array([target_x - current_x, target_y - current_y])
        # Normal vector of car's heading
        vec_normal = np.array(
            [np.cos(np.deg2rad(yaw + 90)), np.sin(np.deg2rad(yaw + 90))])
        # Cross track error
        cross_track_error = np.dot(vec_target, vec_normal)

        # PID control
        self.acc_ep += cross_track_error * dt
        de = (cross_track_error - self.last_ep) / dt if dt > 0 else 0
        self.last_ep = cross_track_error

        # Calculate steering angle - positive when need to turn left
        delta = self.kp * cross_track_error + self.ki * self.acc_ep + self.kd * de

        # Normalize steering angle
        next_delta = utils.angle_norm(delta)
        return next_delta, target
