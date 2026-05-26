import cProfile
from robot_nav.rl_train import main
cProfile.run('main()', 'profile.stats')
