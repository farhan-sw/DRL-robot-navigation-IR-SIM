from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
from a_star import AStarPlanner

import torch
import numpy as np
from robot_nav.SIM_ENV.sim import SIM
import yaml
import math
import matplotlib.pyplot as plt
import argparse

def main(args=None):
    parser = argparse.ArgumentParser(description="Test Hybrid Architecture (A* + RL)")
    parser.add_argument("--random-weights", action="store_true", help="Run with random untrained weights instead of loading the checkpoint.")
    parsed_args = parser.parse_args()
    
    action_dim = 2
    max_action = 1
    state_dim = 185
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Note: By default, we load the trained model.
    # Use --random-weights to test without loading the checkpoint.
    model = CNNTD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        load_model=not parsed_args.random_weights,
        model_name="CNNTD3",
    )

    sim = SIM(world_file="worlds/eval_world.yaml")
    
    with open("robot_nav/eval_points.yaml") as file:
        points = yaml.safe_load(file)
    robot_poses = points["robot"]["poses"]
    robot_goals = points["robot"]["goals"]

    # Initialize A* Planner (Occupancy Grid is built once)
    planner = AStarPlanner(sim, resolution=0.2, margin=0.3)

    print("..............................................")
    print(f"Testing Hybrid Architecture: A* + RL")
    
    for idx in range(len(robot_poses)):
        count = 0
        latest_scan, distance, cos, sin, collision, goal_reached, a, reward = sim.reset(
            robot_state=robot_poses[idx][:3],
            robot_goal=robot_goals[idx],
            random_obstacles=False,
        )
        
        start_pos = (robot_poses[idx][0][0], robot_poses[idx][1][0])
        final_goal = (robot_goals[idx][0][0], robot_goals[idx][1][0])
        
        print(f"\nScenario {idx+1}: Planning A* path from {start_pos} to {final_goal}...")
        global_path = planner.plan(start_pos, final_goal)
        
        if not global_path:
            print("Failed to find global path! Skipping...")
            continue
            
        print(f"Path found with {len(global_path)} waypoints.")
        
        # Visualize the A* path robustly using matplotlib directly (drawn ONCE)
        if hasattr(sim.env, '_env_plot') and hasattr(sim.env._env_plot, 'ax'):
            ax = sim.env._env_plot.ax
            
            # Clear previous scenario's A* drawings if they exist
            if hasattr(sim, 'a_star_lines'):
                for line in sim.a_star_lines:
                    try: line.remove()
                    except: pass
            if hasattr(sim, 'a_star_scatters'):
                for scat in sim.a_star_scatters:
                    try: scat.remove()
                    except: pass
                    
            path_x = [p[0] for p in global_path]
            path_y = [p[1] for p in global_path]
            sim.a_star_lines = ax.plot(path_x, path_y, 'r--', linewidth=2)
            sim.a_star_scatters = [ax.scatter(path_x, path_y, c='blue', s=30, zorder=5)]
            
        current_waypoint_idx = 0
        max_steps = 500
        done = False
        
        while not done and count < max_steps:
            local_goal = global_path[current_waypoint_idx]
            
            robot_state = sim.env.get_robot_state()
            dist_to_local = math.hypot(robot_state[0].item() - local_goal[0], 
                                       robot_state[1].item() - local_goal[1])
            
            # Switch to next waypoint if close enough
            if dist_to_local < 0.4 and current_waypoint_idx < len(global_path) - 1:
                current_waypoint_idx += 1
                local_goal = global_path[current_waypoint_idx]
                
            # Temporarily trick the environment into thinking the local goal is the target
            sim.robot_goal = np.array([[local_goal[0]], [local_goal[1]], [0]])
            sim.env.robot.set_goal(np.array([[local_goal[0]], [local_goal[1]], [0]]))
            
            # Manually recalculate RL inputs for the new local goal
            goal_vector = [
                sim.robot_goal[0].item() - robot_state[0].item(),
                sim.robot_goal[1].item() - robot_state[1].item(),
            ]
            distance = np.linalg.norm(goal_vector).item()
            pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
            cos, sin = sim.cossin(pose_vector, goal_vector)
            
            # Get state and action
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal_reached, a
            )
            action = model.get_action(np.array(state), False)
            a_in = [(action[0] + 1) / 4, action[1]]
            
            # Execute step
            latest_scan, distance, cos, sin, collision, goal_reached, a, reward = sim.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            
            count += 1
            
            # Check if final goal is reached
            dist_to_final = math.hypot(robot_state[0].item() - final_goal[0], 
                                       robot_state[1].item() - final_goal[1])
            if dist_to_final < 0.3:
                goal_reached = True
                
            done = collision or goal_reached
            
        if goal_reached:
            print("Success! Reached final goal.")
        elif collision:
            print("Failed: Collision.")
        else:
            print("Failed: Timeout.")
            
if __name__ == "__main__":
    main()
