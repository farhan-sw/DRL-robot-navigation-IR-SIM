import irsim
import numpy as np
import random

from robot_nav.SIM_ENV.sim_env import SIM_ENV


class SIM(SIM_ENV):
    """
    A simulation environment interface for robot navigation using IRSim.

    This class wraps around the IRSim environment and provides methods for stepping,
    resetting, and interacting with a mobile robot, including reward computation.

    Attributes:
        env (object): The simulation environment instance from IRSim.
        robot_goal (np.ndarray): The goal position of the robot.
    """

    def __init__(self, world_file="robot_world.yaml", disable_plotting=False):
        """
        Initialize the simulation environment.

        Args:
            world_file (str): Path to the world configuration YAML file.
            disable_plotting (bool): If True, disables rendering and plotting.
        """
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        self.disable_plotting = disable_plotting
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal

    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        """
        Perform one step in the simulation using the given control commands.

        Args:
            lin_velocity (float): Linear velocity to apply to the robot.
            ang_velocity (float): Angular velocity to apply to the robot.

        Returns:
            (tuple): Contains the latest LIDAR scan, distance to goal, cosine and sine of angle to goal,
                   collision flag, goal reached flag, applied action, and computed reward.
        """
        self.env.step(action_id=0, action=np.array([[lin_velocity], [ang_velocity]]))
        if not getattr(self, 'disable_plotting', False):
            self.env.render()

        scan = self.env.get_lidar_scan()
        latest_scan = scan["ranges"]

        robot_state = self.env.get_robot_state()
        goal_vector = [
            self.robot_goal[0].item() - robot_state[0].item(),
            self.robot_goal[1].item() - robot_state[1].item(),
        ]
        distance = np.linalg.norm(goal_vector)
        goal = self.env.robot.arrive
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        cos, sin = self.cossin(pose_vector, goal_vector)
        collision = self.env.robot.collision
        action = [lin_velocity, ang_velocity]
        reward = self.get_reward(goal, collision, action, latest_scan, cos)

        return latest_scan, distance, cos, sin, collision, goal, action, reward

    def reset(
        self,
        robot_state=None,
        robot_goal=None,
        random_obstacles=True,
        random_obstacle_ids=None,
    ):
        """
        Reset the simulation environment, optionally setting robot and obstacle states.

        Args:
            robot_state (list or None): Initial state of the robot as a list of [x, y, theta, speed].
            robot_goal (list or None): Goal state for the robot.
            random_obstacles (bool): Whether to randomly reposition obstacles.
            random_obstacle_ids (list or None): Specific obstacle IDs to randomize.

        Returns:
            (tuple): Initial observation after reset, including LIDAR scan, distance, cos/sin,
                   and reward-related flags and values.
        """
        if robot_state is None:
            robot_state = [[random.uniform(1, 9)], [random.uniform(1, 9)], [0]]

        self.env.robot.set_state(
            state=np.array(robot_state),
            init=True,
        )

        if random_obstacles:
            if random_obstacle_ids is None:
                random_obstacle_ids = [i + 1 for i in range(7)]
            self.env.random_obstacle_position(
                range_low=[0, 0, -3.14],
                range_high=[10, 10, 3.14],
                ids=random_obstacle_ids,
                non_overlapping=True,
            )

        if robot_goal is None:
            while True:
                self.env.robot.set_random_goal(
                    obstacle_list=self.env.obstacle_list,
                    init=True,
                    range_limits=[[1, 1, -3.141592653589793], [9, 9, 3.141592653589793]],
                )
                goal_pos = self.env.robot.goal
                dist = np.linalg.norm([
                    goal_pos[0].item() - robot_state[0][0],
                    goal_pos[1].item() - robot_state[1][0]
                ])
                if 1.0 <= dist <= 2.5:
                    break
        else:
            self.env.robot.set_goal(np.array(robot_goal), init=True)
        self.env.reset()
        self.robot_goal = self.env.robot.goal

        action = [0.0, 0.0]
        latest_scan, distance, cos, sin, _, _, action, reward = self.step(
            lin_velocity=action[0], ang_velocity=action[1]
        )
        return latest_scan, distance, cos, sin, False, False, action, reward

    @staticmethod
    def get_reward(goal, collision, action, laser_scan, cos_angle=0.0):
        """
        Calculate the reward for the current step.

        Args:
            goal (bool): Whether the goal has been reached.
            collision (bool): Whether a collision occurred.
            action (list): The action taken [linear velocity, angular velocity].
            laser_scan (list): The LIDAR scan readings.
            cos_angle (float): The cosine of the angle between robot heading and goal direction.

        Returns:
            (float): Computed reward for the current state.
        """
        if goal:
            return 200.0  # High reward for precise docking
        elif collision:
            return -500.0  # Catastrophic penalty for forklift crash
        else:
            # action[0] = lin_vel (-1 to 1), action[1] = ang_vel (-1 to 1)
            # Progress reward: moving towards the goal (cos > 0)
            progress_reward = action[0] * (cos_angle * 2.0)
            
            # Smoothness penalty: Forklifts shouldn't make jerky turns to avoid dropping payload
            spin_penalty = abs(action[1]) * 0.5
            
            # Proximity penalty: Forklifts must maintain a safe distance (0.5m) from racks/walls
            min_lidar_dist = min(laser_scan) if len(laser_scan) > 0 else 10.0
            obstacle_penalty = 0.0
            if min_lidar_dist < 0.5:
                obstacle_penalty = (0.5 - min_lidar_dist) * 2.0
            
            # Base time penalty (relaxed to 0.1 to allow stopping/waiting)
            time_penalty = 0.1
            
            return progress_reward - spin_penalty - obstacle_penalty - time_penalty
