from PathPlanning.planner import Planner
import PathPlanning.utils as utils
import cv2
import numpy as np
import sys
sys.path.append("..")


class PlannerRRTStar(Planner):
    def __init__(self, m, extend_len=20, rewire_radius=50):
        """Initialize with map, step size, and rewiring radius."""
        super().__init__(m)
        self.extend_len = extend_len
        self.rewire_radius = rewire_radius  # Radius for finding nearby nodes

    def _random_node(self, goal, shape):
        """Sample a random node, with 50% chance of choosing the goal."""
        r = np.random.choice(2, 1, p=[0.5, 0.5])
        if r == 1:
            return (float(goal[0]), float(goal[1]))
        else:
            rx = float(np.random.randint(int(shape[1])))
            ry = float(np.random.randint(int(shape[0])))
            return (rx, ry)

    def _nearest_node(self, samp_node):
        """Find the node in the tree closest to the sampled node."""
        min_dist = float('inf')  # Use infinity instead of 99999 for robustness
        min_node = None
        for n in self.ntree:
            dist = utils.distance(n, samp_node)
            if dist < min_dist:
                min_dist = dist
                min_node = n
        return min_node

    def _check_collision(self, n1, n2):
        """Check for obstacles between two nodes using Bresenham's algorithm."""
        n1_ = utils.pos_int(n1)
        n2_ = utils.pos_int(n2)
        line = utils.Bresenham(n1_[0], n2_[0], n1_[1], n2_[1])
        for pts in line:
            if self.map[int(pts[1]), int(pts[0])] < 0.5:
                return True
        return False

    def _steer(self, from_node, to_node, extend_len):
        """Extend from from_node toward to_node by extend_len, checking collisions."""
        vect = np.array(to_node) - np.array(from_node)
        v_len = np.hypot(vect[0], vect[1])
        if v_len == 0:
            return False, None
        v_theta = np.arctan2(vect[1], vect[0])
        if extend_len > v_len:
            extend_len = v_len
        new_node = (from_node[0] + extend_len * np.cos(v_theta),
                    from_node[1] + extend_len * np.sin(v_theta))
        if (new_node[1] < 0 or new_node[1] >= self.map.shape[0] or
            new_node[0] < 0 or new_node[0] >= self.map.shape[1] or
                self._check_collision(from_node, new_node)):
            return False, None
        else:
            return new_node, utils.distance(new_node, from_node)

    def _nearby_nodes(self, node, r):
        """Find all nodes within radius r of the given node."""
        return [n for n in self.ntree if utils.distance(n, node) < r]

    def planning(self, start, goal, extend_len=None, img=None):
        """Plan an optimized path from start to goal using RRT*."""
        if extend_len is None:
            extend_len = self.extend_len

        # Initialize tree and costs
        self.ntree = {start: None}
        self.cost = {start: 0}

        # Grow the tree for a fixed number of iterations
        for it in range(3000):

            if it % 100 == 0:
                print(f"Iteration: {it}")

            samp_node = self._random_node(goal, self.map.shape)
            near_node = self._nearest_node(samp_node)
            new_node, cost = self._steer(near_node, samp_node, extend_len)
            if new_node is False:
                continue

            # Add new node with nearest node as parent
            self.ntree[new_node] = near_node
            self.cost[new_node] = cost + self.cost[near_node]

            # Re-Parenting: Find the best parent among nearby nodes
            nearby = self._nearby_nodes(new_node, self.rewire_radius)
            min_cost = self.cost[new_node]
            min_parent = near_node
            for X in nearby:
                if not self._check_collision(X, new_node):
                    temp_cost = self.cost[X] + utils.distance(X, new_node)
                    if temp_cost < min_cost:
                        min_cost = temp_cost
                        min_parent = X
            if min_parent != near_node:
                self.ntree[new_node] = min_parent
                self.cost[new_node] = min_cost

            # Re-Wiring: Update nearby nodes if new_node offers a lower cost
            for X in nearby:
                if not self._check_collision(new_node, X):
                    temp_cost = self.cost[new_node] + \
                        utils.distance(new_node, X)
                    if temp_cost < self.cost[X]:
                        self.ntree[X] = new_node
                        self.cost[X] = temp_cost

            # Visualization
            if img is not None:
                for n in self.ntree:
                    if self.ntree[n] is None:
                        continue
                    node = self.ntree[n]
                    cv2.line(img, utils.pos_int(n),
                             utils.pos_int(node), (0, 1, 0), 1)
                img_ = img.copy()
                cv2.circle(img_, utils.pos_int(new_node), 5, (0, 0.5, 1), 3)
                img_ = cv2.flip(img_, 0)
                cv2.imshow("Path Planning", img_)
                cv2.waitKey(1)

        # Find the best node to connect to the goal
        potential_goal_nodes = [
            n for n in self.ntree
            if utils.distance(n, goal) < extend_len and not self._check_collision(n, goal)
        ]
        if potential_goal_nodes:
            # Choose the node with the lowest total cost to goal
            goal_node = min(potential_goal_nodes,
                            key=lambda n: self.cost[n] + utils.distance(n, goal))
            path = []
            n = goal_node
            while n is not None:
                path.insert(0, n)
                n = self.ntree[n]
            path.append(goal)  # Add the goal as the final point
        else:
            print("No path found")
            path = []

        return path
