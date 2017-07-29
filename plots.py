import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def compare_scores_plot(*cv_scores):
    for scores, color in zip(cv_scores, sns.color_palette("Set1", n_colors=len(cv_scores))):
        plot_kfold_scores(scores, color=color, plot_immediately=False)
    plt.plot()


def plot_kfold_scores(cv_scores, color='blue', plot_immediately=True):
    plt.scatter(range(len(cv_scores)), cv_scores, c=color)
    plt.axhline(y=np.mean(cv_scores), xmin=0, xmax=len(cv_scores), c=color)
    if plot_immediately:
        plt.show()