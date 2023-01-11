import pickle
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import os

base_dir = os.path.dirname(__file__)
saved_data_dir = os.path.join(base_dir, 'saved_data')

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plot_C_gamma(grid):
    C_range = grid.param_grid['C']
    gamma_range = grid.param_grid['gamma']
    scores = grid.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range))


    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        scores,
        interpolation="nearest",
        cmap=plt.cm.hot,
        norm=MidpointNormalize(vmin=0.2, midpoint=0.92),
    )
    plt.xlabel("gamma")
    plt.ylabel("C")
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title("Validation accuracy")
    plt.show()

with open(os.path.join(saved_data_dir, 'saved_grid_SVM.pkl'), 'rb') as inp:
    plot_C_gamma(pickle.load(inp))