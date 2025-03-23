from PathTracking.controller import Controller
import PathTracking.utils as utils
import sys
import numpy as np
sys.path.append("..")


class ControllerStanleyBicycle(Controller):
    def __init__(self, kp=0.5):
        self.path = None
        self.kp = kp
        # 添加前一幀的橫向誤差，用於計算變化率
        self.prev_cross_track_error = 0
        # 添加平滑參數，減少控制輸出的震盪
        self.prev_delta = 0

    # State: [x, y, yaw, delta, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None

        # Extract State
        x, y, yaw, delta, v, l = info["x"], info["y"], info["yaw"], info["delta"], info["v"], info["l"]

        # 計算前輪位置，這是Stanley控制器的重要參考點
        front_x = x + l*np.cos(np.deg2rad(yaw))
        front_y = y + l*np.sin(np.deg2rad(yaw))

        # 避免在低速或倒車時產生異常
        vf = max(0.1, v / np.cos(np.deg2rad(delta)))

        # 尋找最近的路徑點
        min_idx, min_dist = utils.search_nearest(self.path, (front_x, front_y))
        target = self.path[min_idx]

        # 尋找更前方的目標點，增加前瞻性
        # 尤其是當車輛方向與路徑方向差異大時，這很重要
        lookahead_distance = min(10.0, max(2.0, vf * 0.5))  # 動態前瞻距離
        target_idx = min_idx
        dist_sum = 0

        # 向前尋找符合前瞻距離的點
        for i in range(min_idx, len(self.path)-1):
            dist = np.hypot(self.path[i+1, 0] - self.path[i, 0],
                            self.path[i+1, 1] - self.path[i, 1])
            dist_sum += dist
            if dist_sum >= lookahead_distance:
                target_idx = i + 1
                break

        # 確保索引不超出範圍
        target_idx = min(target_idx, len(self.path)-1)
        target = self.path[target_idx]

        # 計算航向誤差 (theta_e)
        target_yaw = target[2]
        heading_error = np.deg2rad(utils.angle_norm(target_yaw - yaw))

        # 改進橫向誤差的計算方法
        # 使用前輪位置到路徑的投影距離，而不只是簡單的叉積
        # 這確保了當車輛與路徑平行時也能計算正確的橫向距離
        path_direction = np.array(
            [np.cos(np.deg2rad(target_yaw)), np.sin(np.deg2rad(target_yaw))])
        # 路徑法向量 (正交於路徑方向)
        path_normal = np.array(
            [np.cos(np.deg2rad(target_yaw + 90)), np.sin(np.deg2rad(target_yaw + 90))])
        # 前輪到目標點的向量
        front_to_target = np.array([front_x - target[0], front_y - target[1]])

        # 計算橫向誤差 (正值表示車輛在路徑右側，負值表示左側)
        cross_track_error = np.dot(front_to_target, path_normal)

        # 計算縱向誤差 (用於判斷車輛是否在向前行駛)
        longitudinal_error = np.dot(front_to_target, path_direction)

        # 動態調整控制增益
        # 當誤差大時，增加增益以更快接近路徑
        dynamic_kp = self.kp * (1.0 + min(1.0, abs(cross_track_error) / 10.0))

        # 當橫向誤差和航向誤差同向時，需要更積極的控制
        # 例如：車輛偏右且車頭朝右，或車輛偏左且車頭朝左
        if cross_track_error * np.sin(heading_error) > 0:
            dynamic_kp *= 1.5

        # 改進的Stanley控制律
        # 首先，確定橫向誤差項的符號，當車輛在路徑的不同側時應有不同方向的修正
        if longitudinal_error < 0:  # 如果車輛在目標點之前，需要反轉橫向誤差的符號
            cross_track_error = -cross_track_error

        # 計算橫向控制項，限制在合理範圍內
        cross_track_term = np.arctan2(dynamic_kp * cross_track_error, vf)

        # 組合航向誤差和橫向誤差，計算轉向角
        delta_raw = heading_error + cross_track_term

        # 限制轉向角的變化率，避免急轉彎
        delta_change_limit = np.deg2rad(
            max(5.0, min(20.0, vf * 1.0)))  # 動態變化率限制
        delta_raw = np.clip(delta_raw,
                            np.deg2rad(delta) - delta_change_limit,
                            np.deg2rad(delta) + delta_change_limit)

        # 轉換為角度並限制在車輛的轉向範圍內
        next_delta = np.rad2deg(
            np.clip(delta_raw, np.deg2rad(-45), np.deg2rad(45)))

        # 應用平滑因子，減少控制輸出的震盪
        smooth_factor = 0.7
        next_delta = smooth_factor * next_delta + \
            (1 - smooth_factor) * self.prev_delta
        self.prev_delta = next_delta

        # 更新前一幀的橫向誤差
        self.prev_cross_track_error = cross_track_error

        # 輸出調試信息
        if min_idx % 20 == 0:  # 每20幀輸出一次，避免太多輸出
            print(f"\n[Stanley Debug] idx:{min_idx}/{len(self.path)-1}")
            print(
                f"Position: ({x:.1f},{y:.1f}), Front: ({front_x:.1f},{front_y:.1f})")
            print(
                f"Target: ({target[0]:.1f},{target[1]:.1f}), Yaw: {target_yaw:.1f}°")
            print(f"Heading Error: {np.rad2deg(heading_error):.1f}°")
            print(f"Cross-Track Error: {cross_track_error:.2f}")
            print(f"Stanley Term: {np.rad2deg(cross_track_term):.1f}°")
            print(f"Next Steering Angle: {next_delta:.1f}°")

        return next_delta, target
