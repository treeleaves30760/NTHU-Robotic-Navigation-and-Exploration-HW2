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
        self.prev_target_idx = 0  # Keep track of previous target index

    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None

        # Extract State
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]

        # Search Front Target
        min_idx, min_dist = utils.search_nearest(self.path, (x, y))

        # Forward search to ensure we don't go backwards on the path
        if min_idx < self.prev_target_idx:
            min_idx = self.prev_target_idx

        # Calculate lookahead distance - reduce to make the car more responsive
        Ld = self.kp*v + self.Lfc

        # Ensure we have a minimum lookahead distance even at low speeds
        Ld = max(Ld, 5.0)

        # Find target point
        target_idx = min_idx
        for i in range(min_idx, len(self.path)-1):
            dist = np.sqrt((self.path[i+1, 0]-x)**2 + (self.path[i+1, 1]-y)**2)
            if dist > Ld:
                target_idx = i
                break

        # If we're at the end of the path, use the last point
        if target_idx >= len(self.path) - 1:
            target_idx = len(self.path) - 1

        # Save the current target index for next iteration
        self.prev_target_idx = target_idx

        # Get target point
        target = self.path[target_idx]

        # Pure Pursuit Control for Basic Kinematic Model
        # Calculate the angle between the robot heading and the target point
        alpha = np.arctan2(target[1] - y, target[0] - x) - np.deg2rad(yaw)

        # Ensure alpha is within [-pi, pi]
        # More efficient way to normalize angle
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        # Calculate the angular velocity based on pure pursuit
        # For low speeds, maintain a minimum velocity for the calculation
        # Use at least 0.5 for calculation to prevent instability at low speeds
        calc_v = max(v, 0.5)
        next_w = np.rad2deg(2.0 * calc_v * np.sin(alpha) / Ld)

        # Apply a limit to the angular velocity to prevent abrupt turns
        max_w = 30.0  # Reduced maximum angular velocity
        next_w = np.clip(next_w, -max_w, max_w)

        # For very small angles, reduce angular velocity to prevent oscillation
        if abs(np.rad2deg(alpha)) < 5.0:
            next_w = next_w * 0.5

        # Normalize the angular velocity
        next_w = utils.angle_norm(next_w)

        # Print debug information
        # print(f'target idx: {target_idx}/{len(self.path)-1}')
        # print(f'target pos: ({target[0]:.2f}, {target[1]:.2f})')
        # print(f'current pos: ({x:.2f}, {y:.2f}, {yaw:.2f}°)')
        # print(f'alpha angle: {np.rad2deg(alpha):.2f}°')
        # print(f'lookahead distance: {Ld:.2f}')
        # print(f'angular velocity: {next_w:.2f}°/s')
        # print('-' * 40)

        return next_w, target
