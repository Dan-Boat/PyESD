# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:25:24 2022

@author: dboateng
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_style():
    plt.style.use("bmh")
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    mpl.rc('text', usetex=False)
    mpl.rc('font', size=18, family='serif')
    mpl.rc('xtick', labelsize=22)
    mpl.rc('ytick', labelsize=22)
    mpl.rc('legend', fontsize=16)
    mpl.rc('axes', labelsize=22)
    mpl.rc('lines', linewidth=3)

def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot(
        [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--r", linewidth=2
    )
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    extra = plt.Rectangle(
        (0, 0), 0, 0, fc="w", fill=False, edgecolor="none", linewidth=0
    )
    ax.legend([extra], [scores], loc="upper left")
    title = title + "\n Evaluation in {:.2f} seconds".format(elapsed_time)
    ax.set_title(title)
    
    