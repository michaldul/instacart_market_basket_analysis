import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def compare_scores_plot(*cv_scores):
    for scores, color in zip(cv_scores, sns.color_palette("Set1", n_colors=len(cv_scores))):
        plot_kfold_scores(scores, color=color, plot_immediately=False)
    plt.plot()


def plot_kfold_scores(cv_scores, color='blue', plot_immediately=True, y_lim=(.370, .385)):
    plt.scatter(range(len(cv_scores)), cv_scores, c=color)
    plt.axhline(y=np.mean(cv_scores), xmin=0, xmax=len(cv_scores), c=color)
    plt.text(0.76, 0.9, 'mean(score):', transform=plt.gca().transAxes, horizontalalignment='right')
    plt.text(0.77, 0.9, '{:.5f}'.format(np.mean(cv_scores)), transform=plt.gca().transAxes)
    plt.text(0.76, 0.84, 'std(score):', transform=plt.gca().transAxes, horizontalalignment='right')
    plt.text(0.77, 0.84, '{:.5f}'.format(np.std(cv_scores)), transform=plt.gca().transAxes)
    plt.gca().set_ylim(*y_lim)
    if plot_immediately:
        plt.show()
