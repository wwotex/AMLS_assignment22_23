import pickle
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import os
import globals
from sklearn.metrics import confusion_matrix



globals.initialize()

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
    plt.grid()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title("Validation accuracy")
    plt.savefig(os.path.join(globals.image_dir, 'c_gamma_svm.jpg'))
    plt.show()

def plot_confusion_matrix(results, model):
    conf_matrix = confusion_matrix(results[0], results[1])
    
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(os.path.join(globals.image_dir, f'confusion_{model}.jpg'))
    plt.show()

def plot_learning_curve(grid, model, param_name):

    train_scores_mean = grid.cv_results_['mean_train_score']
    train_scores_std = grid.cv_results_['std_train_score']
    test_scores_mean = grid.cv_results_['mean_test_score']
    test_scores_std = grid.cv_results_['std_test_score']
    param_range = grid.param_grid[param_name]

    # plot the validation curve
    plt.title("Validation Curve")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    # plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(os.path.join(globals.image_dir, f'validation_curve_{model}.jpg'))
    plt.show()

def plot_score_parameter(grid, model, param_name):

    # train_scores_mean = grid.cv_results_['mean_train_score']
    # train_scores_std = grid.cv_results_['std_train_score']
    test_scores_mean = grid.cv_results_['mean_test_score']
    test_scores_std = grid.cv_results_['std_test_score']
    param_range = grid.param_grid[param_name]

    # plot the validation curve
    plt.title(f"Performance vs {param}")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    # plt.ylim(0.0, 1.1)
    lw = 2
    # plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    # plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
    # plt.legend(loc="best")
    plt.grid()
    plt.savefig(os.path.join(globals.image_dir, f'{model}_{param}.jpg'))
    plt.show()

if __name__ == "__main__":
    model = 'svm'
    param = 'gamma'
    with open(os.path.join(globals.saved_data_dir, 'saved_grid_SVM.pkl'), 'rb') as inp:
        plot_C_gamma(pickle.load(inp))

    with open(os.path.join(globals.saved_data_dir, 'saved_grid_SVM.pkl'), 'rb') as inp:
        plot_learning_curve(pickle.load(inp), model, param)

    with open(os.path.join(globals.saved_data_dir, 'saved_grid_SVM.pkl'), 'rb') as inp:
        plot_score_parameter(pickle.load(inp), model, param)

    with open(os.path.join(globals.saved_data_dir, 'saved_results_SVM.pkl'), 'rb') as inp:
        results = pickle.load(inp)
        plot_confusion_matrix(results, model)