import torch
import numpy as np
from utils.evaluation import compute_statistics

from matplotlib import pyplot as plt

class Evaluator:
    def __init__(self, config, exp_root):
        self.config = config
        self.exp_root = exp_root

    def evaluate(self, test_data):
        samples_dir = self.exp_root / "samples"
        klds, klcs = [], []

        for filename in sorted(samples_dir.glob("*")):
            samples = torch.load(filename)
            kld, klc = compute_statistics(test_data, samples)
            klds.append(kld)
            klcs.append(klc)

        total = np.array(klds) + np.array(klcs)
        plt.plot(np.array(klds))
        plt.plot(np.array(klcs))
        plt.plot(total)
        plt.show()





