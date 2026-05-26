import sys, os
sys.path.append(os.path.join(os.getcwd(), 'robot_nav'))
from SIM_ENV.sim import SIM
import matplotlib.pyplot as plt

sim1 = SIM("robot_nav/worlds/eval_world.yaml")
sim1.step(0.1, 0.1)
plt.close('all')

sim2 = SIM("robot_nav/worlds/robot_world.yaml")
sim2.step(0.1, 0.1)
print("SUCCESS!")
