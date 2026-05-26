import sys, os
sys.path.append(os.getcwd()+"/robot_nav")
import torch, timeit, numpy as np
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
model = CNNTD3(185, 2, 1, torch.device('cuda'))
class DummyBuffer:
    def sample_batch(self, bs):
        return (np.random.rand(bs, 185), np.random.rand(bs, 2), np.random.rand(bs, 1), np.random.rand(bs, 1), np.random.rand(bs, 185))
buf = DummyBuffer()
print('Train 80 iters:', timeit.timeit('model.train(buf, 80, 64)', globals=globals(), number=10))
