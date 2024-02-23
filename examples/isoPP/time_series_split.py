# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:42:39 2024

@author: dboateng
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.model_selection import KFold, TimeSeriesSplit
from datetime import datetime, timedelta
from matplotlib.dates import YearLocator
import matplotlib.dates as mdates
import pandas as pd 

from pyESD.plot_utils import apply_style

rng = np.random.RandomState(1338)
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
n_splits = 20

# Generate the class/group data
n_points = 400
X = rng.randn(400, 10)

percentiles_classes = [0.1, 0.3, 0.6]
y = np.hstack([[ii] * int(400 * perc) for ii, perc in enumerate(percentiles_classes)])

# Generate uneven groups
group_prior = rng.dirichlet([2] * 10)
groups = np.repeat(np.arange(10), rng.multinomial(400, group_prior))

# Generate datetime indices
start_date = datetime(2024, 1, 1)
datetime_indices = [start_date + timedelta(days=i) for i in range(n_points)]

datetime_indices = pd.date_range(start='1/1/1979', periods=n_points, freq="MS")

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            datetime_indices,
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        datetime_indices, [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        datetime_indices, [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Datetime",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax



path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots"



cvs = [KFold, TimeSeriesSplit]
figname =["kfold.pdf", "timeseries.pdf"]

for i,cv in enumerate(cvs):
    this_cv = cv(n_splits=n_splits)
    
    apply_style(fontsize=22, style=None, linewidth=3)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_cv_indices(this_cv, X, y, groups, ax, n_splits)

    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )
    ax.xaxis.set_major_locator(YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    fig.subplots_adjust(right=0.7)
    plt.savefig(os.path.join(path_to_save, figname[i]), bbox_inches="tight", format="pdf", 
                dpi=300)
plt.show()
