import sys, os
sys.path.append(os.path.join(os.getcwd(), 'robot_nav'))
from SIM_ENV.sim import SIM
sim = SIM("robot_nav/worlds/robot_world.yaml", disable_plotting=True)
sim.reset()
print(sim.env.robot.goal)
print(type(sim.env.robot.goal))
print(sim.env.robot.goal.shape)
print(sim.env.robot.goal[0].item())
