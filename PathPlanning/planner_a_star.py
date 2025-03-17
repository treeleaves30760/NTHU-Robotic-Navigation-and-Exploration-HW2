from PathPlanning.planner import Planner
import PathPlanning.utils as utils
import cv2
import sys
import heapq
sys.path.append("..")


class PlannerAStar(Planner):
    def __init__(self, m, inter=10):
        super().__init__(m)
        self.inter = inter
        self.initialize()

    def initialize(self):
        self.queue = []
        self.parent = {}
        self.h = {}  # Distance from start to node
        self.g = {}  # Distance from node to goal
        self.goal_node = None
        self.closed_set = set()

    def planning(self, start=(100, 200), goal=(375, 520), inter=None, img=None):
        if inter is None:
            inter = self.inter
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        # Initialize
        self.initialize()

        # Calculate h value for start node BEFORE using it
        self.g[start] = 0
        self.h[start] = utils.distance(start, goal)

        # Now we can use self.h[start] safely
        heapq.heappush(self.queue, (self.h[start], start))
        self.parent[start] = None
        while self.queue:
            # Get the node with the lowest f value from the priority queue
            current_f, current = heapq.heappop(self.queue)

            # If we reach the goal, break the loop
            if utils.distance(current, goal) < inter:
                self.goal_node = current
                break

            # Add the current node to the closed set
            self.closed_set.add(current)

            # Generate neighbors (in 8 directions)
            neighbors = []
            for dx in [-inter, 0, inter]:
                for dy in [-inter, 0, inter]:
                    if dx == 0 and dy == 0:
                        continue  # Skip the current node

                    neighbor = (current[0] + dx, current[1] + dy)

                    # Skip if neighbor is in closed set
                    if neighbor in self.closed_set:
                        continue

                    # Skip if neighbor is an obstacle (if map is available)
                    if self.map is not None:
                        x, y = neighbor
                        if (x < 0 or x >= self.map.shape[1] or
                            y < 0 or y >= self.map.shape[0] or
                                self.map[y, x] < 0.5):
                            continue

                    # Check for collision using Bresenham's line algorithm (if map is available)
                    if self.map is not None:
                        has_collision = False
                        for p in utils.Bresenham(current[0], neighbor[0], current[1], neighbor[1]):
                            if (p[0] < 0 or p[0] >= self.map.shape[1] or
                                p[1] < 0 or p[1] >= self.map.shape[0] or
                                    self.map[p[1], p[0]] < 0.5):
                                has_collision = True
                                break
                        if has_collision:
                            continue

                    # Calculate g value for this neighbor
                    tentative_g = self.g[current] + \
                        utils.distance(current, neighbor)

                    # If this neighbor is not in the queue or we found a better path
                    if neighbor not in self.g or tentative_g < self.g[neighbor]:
                        self.parent[neighbor] = current
                        self.g[neighbor] = tentative_g
                        self.h[neighbor] = utils.distance(neighbor, goal)
                        f = tentative_g + self.h[neighbor]

                        # Add to queue or update priority
                        heapq.heappush(self.queue, (f, neighbor))

        # Extract path
        path = []
        p = self.goal_node
        if p is None:
            return path
        while (True):
            path.insert(0, p)
            if self.parent[p] is None:
                break
            p = self.parent[p]
        if path[-1] != goal:
            path.append(goal)
        return path
