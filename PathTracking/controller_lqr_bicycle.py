from PathTracking.controller import Controller
import PathTracking.utils as utils
import sys
import numpy as np
sys.path.append("..")


class ControllerLQRBicycle(Controller):
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
        x, y, yaw, delta, v, l, dt = info["x"], info["y"], info["yaw"], info["delta"], info["v"], info["l"], info["dt"]
        yaw = utils.angle_norm(yaw)

        # Search Nesrest Target
        min_idx, min_dist = utils.search_nearest(self.path, (x, y))
        target = self.path[min_idx]
        target[2] = utils.angle_norm(target[2])

        # 採用更前方的目標點來提高前瞻性
        lookahead_distance = 2

        lookahead_idx = min_idx
        accumulated_dist = 0.0

        # 限制搜索範圍，避免超出路徑邊界
        max_search_idx = min(min_idx + 30, len(self.path) - 1)

        for i in range(min_idx, max_search_idx):
            if i + 1 >= len(self.path):
                break

            delta_dist = np.hypot(self.path[i+1, 0] - self.path[i, 0],
                                  self.path[i+1, 1] - self.path[i, 1])
            accumulated_dist += delta_dist
            if accumulated_dist >= lookahead_distance:
                lookahead_idx = i + 1
                break

        # 確保索引不會超出範圍
        lookahead_idx = min(lookahead_idx, len(self.path) - 1)

        if lookahead_idx > min_idx:
            target = self.path[lookahead_idx]
            target[2] = utils.angle_norm(target[2])

        # 如果非常接近終點，則直接使用終點作為目標
        if min_idx >= len(self.path) - 3:
            target = self.path[-1]
            target[2] = utils.angle_norm(target[2])

        # LQR Control for Bicycle Kinematic Model
        tx, ty, tyaw = target[0], target[1], target[2]

        # 計算前輪位置（自行車模型需考慮前輪位置）
        front_x = x + l * np.cos(np.deg2rad(yaw))
        front_y = y + l * np.sin(np.deg2rad(yaw))

        # 計算橫向誤差
        # 目標路徑的法向量
        path_normal = np.array(
            [np.cos(np.deg2rad(tyaw + 90)), np.sin(np.deg2rad(tyaw + 90))])
        # 前輪到目標點的向量
        vec_error = np.array([front_x - tx, front_y - ty])
        # 橫向誤差（帶符號）
        e = np.dot(vec_error, path_normal)

        # 航向誤差（使用弧度計算）- 確保使用弧度單位
        th_e = np.deg2rad(utils.angle_norm(yaw - tyaw))

        # 如果速度太低，增加最小速度以避免奇異性
        calc_v = max(v, 0.5)

        # 計算前饋控制項 - 預測下一個目標點的曲率
        # 安全獲取未來索引，避免越界
        future_idx = min(lookahead_idx + 3, len(self.path) - 1)
        next_target = self.path[future_idx]
        future_curvature = next_target[3]  # 曲率在路徑的第4列

        # 增加前饋控制的預測性，考慮多個未來點的曲率
        if future_idx < len(self.path) - 3:
            future_curvature_2 = self.path[future_idx + 2, 3]
            future_curvature = 0.7 * future_curvature + 0.3 * future_curvature_2

        feedforward_delta = np.arctan2(l * future_curvature, 1.0)
        feedforward_delta = np.rad2deg(feedforward_delta)

        # 建立狀態空間模型：x = [e, dot_e, th_e, dot_th_e]
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        A[0, 1] = dt
        A[1, 2] = calc_v
        A[2, 2] = 1.0
        A[2, 3] = dt

        B = np.zeros((4, 1))
        B[3, 0] = calc_v / l  # 自行車模型的控制輸入影響

        # 根據車速和誤差動態調整控制增益
        Q_adjusted = self.Q.copy()

        # 優化速度因子的計算
        # 低速時需要更積極的控制，高速時則需要更平滑的控制
        speed_factor = 8.0 / (calc_v + 0.5)
        speed_factor = np.clip(speed_factor, 0.8, 8.0)

        # 誤差大時提供更強的控制力度
        error_magnitude = abs(e)
        error_factor = min(5.0, 1.0 + error_magnitude / 5.0)

        # 誤差變化率因子 - 誤差增加時提高控制力度
        error_rate = (e - self.pe) / dt if dt > 0 else 0
        rate_factor = 1.0
        if abs(error_rate) > 0.5:  # 如果誤差在增加
            rate_factor = 1.5 if error_rate * e > 0 else 1.0  # 誤差增大時加強控制

        # 航向誤差因子 - 航向偏差大時提高轉向響應
        heading_factor = min(3.0, 1.0 + abs(th_e) * 5.0)

        # 調整各項權重
        Q_adjusted[0, 0] = 12.0 * error_factor * speed_factor  # 橫向誤差權重
        Q_adjusted[1, 1] = 3.0 * rate_factor  # 橫向誤差變化率權重
        Q_adjusted[2, 2] = 8.0 * heading_factor  # 航向誤差權重
        Q_adjusted[3, 3] = 2.0  # 航向誤差變化率權重

        # 動態調整R - 不同場景下調整控制成本
        R_base = 250.0  # 基礎控制成本

        # 高速時增加控制成本，避免急轉彎
        if calc_v > 4.0:
            R_factor = 1.0 + (calc_v - 4.0) / 3.0
        # 低速時降低控制成本，提高響應速度
        else:
            R_factor = max(0.2, 1.0 - (4.0 - calc_v) / 5.0)

        # 誤差小時增加平滑度（增加R）
        if abs(e) < 2.0 and abs(th_e) < np.deg2rad(10):
            R_factor *= 1.5

        R_adjusted = np.eye(1) * R_base * R_factor

        # 解 LQR 問題
        K, P, _ = self._dlqr(A, B, Q_adjusted, R_adjusted)

        # 狀態向量
        x_state = np.zeros((4, 1))
        x_state[0, 0] = e
        x_state[1, 0] = (e - self.pe) / dt if dt > 0 else 0
        x_state[2, 0] = th_e
        x_state[3, 0] = (th_e - self.pth_e) / dt if dt > 0 else 0

        # 計算轉向角 - 結合LQR反饋和前饋項
        lqr_steering = -float(K @ x_state)

        # 根據速度和誤差動態調整前饋影響
        ff_gain_base = 0.7  # 基礎前饋增益

        # 高速時增加前饋增益，以提高預測性
        ff_gain = ff_gain_base * (1.0 + min(1.0, calc_v / 6.0))

        # 誤差小時增加前饋控制的比重
        if abs(e) < 3.0 and abs(th_e) < np.deg2rad(15):
            ff_gain *= 1.3

        # 組合控制輸出
        steering = lqr_steering + ff_gain * feedforward_delta

        # 動態調整最大轉向角度
        max_steer_base = 45.0
        if calc_v < 2.0:  # 低速時允許較大轉向角
            max_steer = max_steer_base
        else:  # 高速時限制轉向角
            # 更溫和的速度衰減曲線
            max_steer = max_steer_base - (calc_v - 2.0) * 1.2
            max_steer = np.clip(max_steer, 15.0, max_steer_base)

        # 限制轉向角範圍
        next_delta = np.clip(steering, -max_steer, max_steer)

        # 根據誤差大小提供增強轉向響應
        if abs(e) > 5.0:  # 非終點區域才增強轉向
            # 誤差非常大時，增強轉向力度
            boost_factor = min(1.5, 1.0 + (abs(e) - 5.0) / 10.0)
            next_delta *= boost_factor
            next_delta = np.clip(next_delta, -max_steer, max_steer)

        # 提供最小轉向響應 - 當誤差存在但轉向角太小時
        min_response_threshold = 3.0  # 觸發最小響應的誤差閾值
        if abs(e) > min_response_threshold and abs(next_delta) < 3.0:
            # 根據誤差方向提供最小轉向
            min_steer = 3.0 * np.sign(e)
            # 平滑過渡
            blend_factor = min(1.0, (abs(e) - min_response_threshold) / 2.0)
            next_delta = next_delta * \
                (1 - blend_factor) + min_steer * blend_factor

        # 轉向角變化率限制 - 避免突然變化
        max_delta_change = 5.0 + calc_v * 0.5  # 動態調整變化率限制

        if abs(next_delta - delta) > max_delta_change:
            # 限制每一步的轉向角變化
            next_delta = delta + np.sign(next_delta - delta) * max_delta_change

        # 更新誤差記錄
        self.pe = e
        self.pth_e = th_e

        # 輸出調試信息
        debug_freq = 20  # 每20幀輸出一次，避免過多輸出
        if min_idx % debug_freq == 0:
            print("\n---------- LQR BICYCLE DEBUG ----------")
            print(f"Current Position: ({x:.2f}, {y:.2f}), Yaw: {yaw:.2f}°")
            print(f"Target Position: ({tx:.2f}, {ty:.2f}), Yaw: {tyaw:.2f}°")
            print(
                f"Lookahead Target: ({target[0]:.2f}, {target[1]:.2f}), Yaw: {target[2]:.2f}°")
            print(f"Front Wheel: ({front_x:.2f}, {front_y:.2f})")
            print(f"Cross-track Error: {e:.2f}")
            print(f"Heading Error: {np.rad2deg(th_e):.2f}°")
            print(f"Index: {min_idx}/{len(self.path)-1}")
            print(f"Vehicle Speed: {v:.2f}, Wheelbase: {l:.2f}")
            print(
                f"Q factors: speed={speed_factor:.2f}, error={error_factor:.2f}, rate={rate_factor:.2f}, heading={heading_factor:.2f}")
            print(f"R factor: {R_factor:.2f}")
            print(f"LQR Gain K: {K.flatten()}")
            print(
                f"State Vector: [{x_state[0,0]:.2f}, {x_state[1,0]:.2f}, {x_state[2,0]:.2f}, {x_state[3,0]:.2f}]")
            print(
                f"Feedforward Steering: {feedforward_delta:.2f}° (gain: {ff_gain:.2f})")
            print(f"LQR Steering: {lqr_steering:.2f}°")
            print(f"Combined Steering: {steering:.2f}°")
            print(f"Final Steering Angle: {next_delta:.2f}°")
            print(f"Current Delta: {delta:.2f}°")
            print(f"Max Allowed Steering: {max_steer:.2f}°")
            print("--------------------------------------")

        return next_delta, target

    def _dlqr(self, A, B, Q, R):
        """求解離散時間 LQR 控制器。
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """
        # 解 Riccati 方程式
        P = self._solve_DARE(A, B, Q, R)

        # 計算 LQR 增益
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        eigvals = np.linalg.eigvals(A - B @ K)

        return K, P, eigvals
