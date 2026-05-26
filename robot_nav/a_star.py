import numpy as np
import heapq
import math

class AStarPlanner:
    def __init__(self, sim_env, resolution=0.2, margin=0.3):
        self.sim = sim_env
        self.resolution = resolution
        self.margin = margin
        self.grid_size_x = int(10 / resolution)
        self.grid_size_y = int(10 / resolution)
        self.grid = np.zeros((self.grid_size_x, self.grid_size_y), dtype=bool)
        self._build_grid()

    def _build_grid(self):
        print("Building Occupancy Grid for A*...")
        orig_state = self.sim.env.robot.state.copy()
        
        # Build raw grid
        raw_grid = np.zeros((self.grid_size_x, self.grid_size_y), dtype=bool)
        for x_idx in range(self.grid_size_x):
            for y_idx in range(self.grid_size_y):
                x = x_idx * self.resolution + (self.resolution / 2)
                y = y_idx * self.resolution + (self.resolution / 2)
                
                self.sim.env.robot.set_state(np.array([[x], [y], [0]]), init=False)
                
                # Check collision with any obstacle
                col = any(self.sim.env.robot.check_collision(obs) for obs in self.sim.env.obstacle_list)
                if col:
                    raw_grid[x_idx, y_idx] = True
                    
        # Apply margin (dilation)
        margin_cells = int(self.margin / self.resolution)
        for x_idx in range(self.grid_size_x):
            for y_idx in range(self.grid_size_y):
                if raw_grid[x_idx, y_idx]:
                    for mx in range(max(0, x_idx - margin_cells), min(self.grid_size_x, x_idx + margin_cells + 1)):
                        for my in range(max(0, y_idx - margin_cells), min(self.grid_size_y, y_idx + margin_cells + 1)):
                            self.grid[mx, my] = True

        self.sim.env.robot.set_state(orig_state, init=False)
        print("Grid built.")

    def heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def get_neighbors(self, node):
        dirs = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        neighbors = []
        for d in dirs:
            nx, ny = node[0] + d[0], node[1] + d[1]
            if 0 <= nx < self.grid_size_x and 0 <= ny < self.grid_size_y:
                if not self.grid[nx, ny]:
                    neighbors.append((nx, ny))
        return neighbors

    def plan(self, start, goal):
        """
        start, goal: (x, y) coordinates
        Returns: list of (x, y) waypoints
        """
        start_node = (int(start[0] / self.resolution), int(start[1] / self.resolution))
        goal_node = (int(goal[0] / self.resolution), int(goal[1] / self.resolution))
        
        # Ensure start and goal are within bounds
        start_node = (min(max(0, start_node[0]), self.grid_size_x-1), min(max(0, start_node[1]), self.grid_size_y-1))
        goal_node = (min(max(0, goal_node[0]), self.grid_size_x-1), min(max(0, goal_node[1]), self.grid_size_y-1))

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_node or self.heuristic(current, goal_node) <= 1.0:
                path = []
                while current in came_from:
                    path.append((current[0] * self.resolution + self.resolution/2, 
                                 current[1] * self.resolution + self.resolution/2))
                    current = came_from[current]
                path.reverse()
                path.append((goal[0], goal[1]))
                return self.smooth_path(path)

            for neighbor in self.get_neighbors(current):
                cost = math.hypot(neighbor[0]-current[0], neighbor[1]-current[1])
                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score, neighbor))

        return [] # No path found
        
    def smooth_path(self, path):
        # A simple path smoothing to remove unnecessary waypoints
        if len(path) <= 2:
            return path
        smoothed_path = [path[0]]
        for i in range(1, len(path) - 1):
            p_prev = smoothed_path[-1]
            p_curr = path[i]
            p_next = path[i+1]
            # Keep waypoint if it represents a turn
            if math.hypot(p_curr[0]-p_prev[0], p_curr[1]-p_prev[1]) > 0.0 and \
               math.hypot(p_next[0]-p_curr[0], p_next[1]-p_curr[1]) > 0.0:
               
               angle1 = math.atan2(p_curr[1]-p_prev[1], p_curr[0]-p_prev[0])
               angle2 = math.atan2(p_next[1]-p_curr[1], p_next[0]-p_curr[0])
               if abs(angle1 - angle2) > 0.1:
                   smoothed_path.append(p_curr)
                   
        smoothed_path.append(path[-1])
        return smoothed_path
