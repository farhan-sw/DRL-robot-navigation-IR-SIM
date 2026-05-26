import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'robot_nav'))

import argparse
from rl_train import main

sys.argv = ['rl_train.py']

import cProfile
import pstats
import threading
import time

def run():
    try:
        main()
    except Exception as e:
        print(e)

profiler = cProfile.Profile()
profiler.enable()

def stop():
    time.sleep(5)
    profiler.disable()
    profiler.dump_stats('profile.stats')
    print("DUMPED STATS")
    os._exit(0)

t = threading.Thread(target=stop)
t.start()

run()
