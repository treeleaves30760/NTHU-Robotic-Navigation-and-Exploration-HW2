from PathTracking.controller import Controller
import PathTracking.utils as utils
import sys
import numpy as np
sys.path.append("..")


class ControllerLQRBasic(Controller):
    def __init__(self, Q=np.eye(4), R=np.eye(1)):
        self.path = None
        self.Q = Q
        self.Q[0, 0] = 1
        self.Q[1, 1] = 1
        self.Q[2, 2] = 1
        self.Q[3, 3] = 1
        self.R = R*5000
        self.pe = 0
        self.pth_e = 0

    def set_path(self, path):
        super().set_path(path)
        self.pe = 0
        self.pth_e = 0

    # Discrete-time Algebra Riccati Equation (DARE)
    def _solve_DARE(self, A, B, Q, R, max_iter=150, eps=0.01):
        P = Q.copy()
        for i in range(max_iter):
            temp = np.linalg.inv(R + B.T @ P @ B)
            Pn = A.T @ P @ A - A.T @ P @ B @ temp @ B.T @ P @ A + Q
            if np.abs(Pn - P).max() < eps:
                break
            P = Pn
        return Pn

    # State: [x, y, yaw, delta, v, l, dt]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None

        # Extract State
        x, y, yaw, v, dt = info["x"], info["y"], info["yaw"], info["v"], info["dt"]
        yaw = utils.angle_norm(yaw)

        # Search Nesrest Target
        min_idx, min_dist = utils.search_nearest(self.path, (x, y))
        target = self.path[min_idx]
        target[2] = utils.angle_norm(target[2])

        # LQR Control for Basic Kinematic Model
        tx, ty, tyaw = target[0], target[1], target[2]

        # Calculate error between current state and reference path
        e = min_dist  # lateral error
        th_e = utils.angle_norm(yaw - tyaw)  # heading error

        # State space model for basic kinematic model
        # x = [e, dot_e, th_e, dot_th_e]
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        A[0, 1] = dt
        A[1, 2] = v
        A[2, 2] = 1.0
        A[2, 3] = dt

        B = np.zeros((4, 1))
        B[3, 0] = v

        # Calculate LQR gain
        K, _, _ = self._dlqr(A, B, self.Q, self.R)

        # State vector
        x = np.zeros((4, 1))
        x[0, 0] = e
        x[1, 0] = (e - self.pe) / dt
        x[2, 0] = th_e
        x[3, 0] = (th_e - self.pth_e) / dt

        # Calculate control input (angular velocity)
        next_w = -float(K @ x)

        # Store errors for next iteration
        self.pe = e
        self.pth_e = th_e

        return next_w, target

    def _dlqr(self, A, B, Q, R):
        """Solve the discrete time LQR controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """
        # Solve the Riccati equation
        P = self._solve_DARE(A, B, Q, R)

        # Compute the LQR gain
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        eigvals = np.linalg.eigvals(A - B @ K)

        return K, P, eigvals
