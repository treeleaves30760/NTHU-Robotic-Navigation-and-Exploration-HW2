import argparse
import numpy as np
import cv2
from Simulation.utils import ControlState
from Simulation.simulator_map import SimulatorMap
from PathPlanning.cubic_spline import *

##############################
# Global Variables
##############################
pose = None
nav_pos = None
way_points = None
path = None
m_cspace = None
set_controller_path = False

##############################
# Navigation
##############################
# Mouse Click Callback


def mouse_click(event, x, y, flags, param):
    global pose, nav_pos, way_points, path, m_cspace, set_controller_path
    if event == cv2.EVENT_LBUTTONUP:
        nav_pos_new = (x, m.shape[0]-y)
        if m_cspace[nav_pos_new[1], nav_pos_new[0]] > 0.5:
            way_points = planner.planning((pose[0], pose[1]), nav_pos_new, 20)
            if len(way_points) > 1:
                nav_pos = nav_pos_new
                path = np.array(cubic_spline_2d(way_points, interval=4))
                set_controller_path = True


def pos_int(p):
    return (int(p[0]), int(p[1]))


def render_path(img, nav_pos, way_points, path):
    cv2.circle(img, nav_pos, 5, (0.5, 0.5, 1.0), 3)
    for i in range(len(way_points)):    # Draw Way Points
        cv2.circle(img, pos_int(way_points[i]), 3, (1.0, 0.4, 0.4), 1)
    for i in range(len(path)-1):    # Draw Interpolating Curve
        cv2.line(img, pos_int(path[i]), pos_int(path[i+1]), (1.0, 0.4, 0.4), 1)
    return img


def navigation(args, simulator, controller, planner, start_pose=(100, 200, 0)):
    global pose, nav_pos, way_points, path, set_controller_path
    # Initialize
    window_name = "Known Map Navigation Demo"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)
    simulator.init_pose(start_pose)
    command = ControlState(args.simulator, None, None)
    pose = start_pose
    collision_count = 0
    goal_reached = False
    # Main Loop
    while (True):
        # Update State
        simulator.step(command)
        pose = (simulator.state.x, simulator.state.y, simulator.state.yaw)
        print("\r", simulator, "| Goal:", nav_pos,
              "| Goal Reached: ", goal_reached, end="\t")

        if set_controller_path:
            controller.set_path(path)
            set_controller_path = False
            goal_reached = False
            print("New Path Set !!")

        if path is not None and collision_count == 0:

            # 檢查是否已到達目標
            if nav_pos is not None and not goal_reached:
                distance_to_goal = np.sqrt(
                    (simulator.state.x - nav_pos[0])**2 + (simulator.state.y - nav_pos[1])**2)
                goal_reached = distance_to_goal <= 10.0

            if goal_reached:
                # 到達目標，根據目前狀態停下車輛，考慮慣性
                if args.simulator == "basic":
                    command = ControlState(args.simulator, 0, 0)
                    next_w = 0
                    next_v = 0
                elif args.simulator == "diff_drive":
                    command = ControlState(args.simulator, 0, 0)
                    next_lw = 0
                    next_rw = 0
                elif args.simulator == "bicycle":
                    # 由於車輛有慣性，因此需要快速減速，並確保到0，並避免後退
                    print(f'simulator.state.v: {simulator.state.v}')
                    if simulator.state.v > 2:
                        command = ControlState(args.simulator, -3.0, 0)
                    elif simulator.state.v < -1:
                        command = ControlState(args.simulator, 10, 0)
                    else:
                        command = ControlState(args.simulator, 0, 0)

                # 清除目標點
                nav_pos = None
                way_points = None
                path = None

            else:
                # TODO: Planning and Controlling
                if args.simulator == "basic":
                    # Get control from controller
                    info = {
                        "x": simulator.state.x,
                        "y": simulator.state.y,
                        "yaw": simulator.state.yaw,
                        "v": simulator.state.v,
                        "w": simulator.state.w,
                        "dt": simulator.dt
                    }
                    next_w, target = controller.feedback(info)
                    next_v = 30.0  # constant velocity
                    command = ControlState("basic", next_v, next_w)
                elif args.simulator == "diff_drive":
                    # Get control from controller
                    info = {
                        "x": simulator.state.x,
                        "y": simulator.state.y,
                        "yaw": simulator.state.yaw,
                        "v": simulator.state.v,
                        "w": simulator.state.w,
                        "dt": simulator.dt
                    }
                    next_w, target = controller.feedback(info)
                    next_v = 30.0  # constant velocity
                    # Convert to differential drive control
                    wheel_base = 15.0  # approximate wheel base
                    next_lw = (next_v - wheel_base * next_w / 2)
                    next_rw = (next_v + wheel_base * next_w / 2)
                    command = ControlState("diff_drive", next_lw, next_rw)
                elif args.simulator == "bicycle":
                    # Get control from controller
                    info = {
                        "x": simulator.state.x,
                        "y": simulator.state.y,
                        "yaw": simulator.state.yaw,
                        "v": simulator.state.v,
                        "w": simulator.state.w,
                        "delta": simulator.cstate.delta,
                        "l": simulator.l,
                        "dt": simulator.dt
                    }
                    next_delta, target = controller.feedback(info)
                    next_a = 1.0 if simulator.state.v < 10.0 else 0.0  # simple acceleration control
                    command = ControlState("bicycle", next_a, next_delta)
                else:
                    exit()
        elif goal_reached:
            # 到達目標，根據目前狀態停下車輛，考慮慣性
            if args.simulator == "basic":
                command = ControlState(args.simulator, 0, 0)
                next_w = 0
                next_v = 0
            elif args.simulator == "diff_drive":
                command = ControlState(args.simulator, 0, 0)
                next_lw = 0
                next_rw = 0
            elif args.simulator == "bicycle":
                # 由於車輛有慣性，因此需要快速減速，並確保到0，並避免後退
                print(f'simulator.state.v: {simulator.state.v}')
                if simulator.state.v > 0.1:
                    command = ControlState(args.simulator, -2.0, 0)
                elif simulator.state.v < 0.1:
                    command = ControlState(args.simulator, 2, 0)
                else:
                    command = ControlState(args.simulator, 0, 0)

        else:
            command = None

        print(f'command: {command}')
        _, info = simulator.step(command)
        # Collision Handling
        if info["collision"]:
            collision_count = 1
            # TODO: Collision Handling
        if collision_count != 0:
            # Go back based on the simulator
            collision_count += 1
            if args.simulator == "basic":
                command = ControlState("basic", -30.0, 0)
            elif args.simulator == "diff_drive":
                command = ControlState("diff_drive", -30.0, 0)
            elif args.simulator == "bicycle":
                command = ControlState("bicycle", -50.0, 0)
            collision_count += 1
        if collision_count > 3:
            # Replan the path
            print(f'Pose: {pose}')
            way_points = planner.planning((pose[0], pose[1]), nav_pos, 20)

            # Check if the path is valid
            if len(way_points) > 1:
                path = np.array(cubic_spline_2d(way_points, interval=4))
                set_controller_path = True
                collision_count = 0
            else:
                # Go back more based on the simulator
                if args.simulator == "basic":
                    command = ControlState("basic", -30.0, 0)
                elif args.simulator == "diff_drive":
                    command = ControlState("diff_drive", -30.0, 0)
                elif args.simulator == "bicycle":
                    command = ControlState("bicycle", -50.0, 0)

        # Render Path
        img = simulator.render()
        if nav_pos is not None and way_points is not None:
            img = render_path(img, nav_pos, way_points, path)

        img = cv2.flip(img, 0)
        cv2.imshow(window_name, img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            simulator.init_state(start_pose)
        if k == 27:
            print()
            break


if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulator", type=str,
                        default="diff_drive", help="diff_drive/bicycle")
    parser.add_argument("-c", "--controller", type=str,
                        default="pure_pursuit", help="pid/pure_pursuit/stanley/lqr")
    parser.add_argument("-p", "--planner", type=str,
                        default="a_star", help="a_star/rrt/rrt_star")
    parser.add_argument("-m", "--map", type=str,
                        default="Maps/map1.png", help="image file name")
    args = parser.parse_args()

    # Read Map
    img = cv2.flip(cv2.imread(args.map), 0)
    img[img > 128] = 255
    img[img <= 128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.
    m_cspace = 1-cv2.dilate(1-m, np.ones((40, 40)))  # Configuration-Space

    # Select Simulator, Controller, and Planner
    try:
        # Simulator / Controller
        if args.simulator == "basic":
            from Simulation.simulator_basic import SimulatorBasic
            simulator = SimulatorMap(
                SimulatorBasic, m=m, l=9, wu=7, wv=3, car_w=16, car_f=13, car_r=7)
            if args.controller == "pid":
                from PathTracking.controller_pid_basic import ControllerPIDBasic as Controller
                controller = Controller()
            elif args.controller == "pure_pursuit":
                from PathTracking.controller_pure_pursuit_basic import ControllerPurePursuitBasic as Controller
                controller = Controller(Lfc=1)
            elif args.controller == "lqr":
                from PathTracking.controller_lqr_basic import ControllerLQRBasic as Controller
                controller = Controller()
            else:
                raise NameError("Unknown controller!!")
        elif args.simulator == "diff_drive":
            from Simulation.simulator_differential_drive import SimulatorDifferentialDrive
            simulator = SimulatorMap(
                SimulatorDifferentialDrive, m=m, l=9, wu=7, wv=3, car_w=16, car_f=13, car_r=7)
            if args.controller == "pid":
                from PathTracking.controller_pid_basic import ControllerPIDBasic as Controller
                controller = Controller()
            elif args.controller == "pure_pursuit":
                from PathTracking.controller_pure_pursuit_basic import ControllerPurePursuitBasic as Controller
                controller = Controller(Lfc=1)
            elif args.controller == "lqr":
                from PathTracking.controller_lqr_basic import ControllerLQRBasic as Controller
                controller = Controller()
            else:
                raise NameError("Unknown controller!!")
        elif args.simulator == "bicycle":
            from Simulation.simulator_bicycle import SimulatorBicycle
            simulator = SimulatorMap(
                SimulatorBicycle, m=m, l=20, d=5, wu=5, wv=2, car_w=14, car_f=25, car_r=5)
            if args.controller == "pid":
                from PathTracking.controller_pid_bicycle import ControllerPIDBicycle as Controller
                controller = Controller()
            elif args.controller == "pure_pursuit":
                from PathTracking.controller_pure_pursuit_bicycle import ControllerPurePursuitBicycle as Controller
                controller = Controller(Lfc=1)
            elif args.controller == "stanley":
                from PathTracking.controller_stanley_bicycle import ControllerStanleyBicycle as Controller
                controller = Controller()
            elif args.controller == "lqr":
                from PathTracking.controller_lqr_bicycle import ControllerLQRBicycle as Controller
                controller = Controller()
            else:
                raise NameError("Unknown controller!!")
        else:
            raise NameError("Unknown simulator!!")
        # Planner
        if args.planner == "a_star":
            from PathPlanning.planner_a_star import PlannerAStar as Planner
        elif args.planner == "rrt":
            from PathPlanning.planner_rrt import PlannerRRT as Planner
        elif args.planner == "rrt_star":
            from PathPlanning.planner_rrt_star import PlannerRRTStar as Planner
        else:
            print("Unknown planner !!")
            exit(0)
        planner = Planner(m_cspace)
    except:
        raise

    navigation(args, simulator, controller, planner)
