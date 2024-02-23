# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:26:04 2024

@author: dboateng
"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


from pyESD.plot_utils import apply_style, correlation_data, count_predictors
from pyESD.plot import correlation_heatmap


from read_data import *
from predictor_setting import *

from pyESD.plot import boxplot
from pyESD.plot_utils import *

path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots"
path_to_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/model_selection"


regressors = ["LassoLarsCV", "RidgeCV","ARD", "BayesianRidge","MLP", 
              "RandomForest", "ExtraTree", "Bagging", "XGBoost", "AdaBoost"]


colors = [grey, purple, "#00FDFF", tomato, skyblue, lightgreen, gold,
                   magenta, orange, blue]




#get data 

mae_df = boxplot_data(regressors=regressors, stationnames=stationnames,
                      path_to_data=path_to_data, filename="validation_score_", 
                      varname="test_rmse")


apply_style(fontsize=22, style=None, linewidth=2)

fig, ax = plt.subplots(1,1, figsize=(15,12)) 

# sns.violinplot(x=' ', y='', data=df, palette='pastel', width=0.7, alpha=0.5, inner_kws=dict(box_width=18, whis_width=2),)
vio = sns.violinplot(data=mae_df, dodge=False,
                    palette=colors, alpha=0.8,
                    width=0.7, inner_kws=dict(box_width=14, whis_width=2, color="black"),
                    bw_adjust=0.67, ax=ax)


ax.tick_params(axis='x', rotation=45)

# Adjust the width of the violin plot to show only half
# for violin in vio.collections:
#     bbox = violin.get_paths()[0].get_extents()
#     x0, y0, width, height = bbox.bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=vio.transData))

# Adjust the width of the violin plot to show only half

for violin in vio.collections:
    paths = violin.get_paths()
    for path in paths:
        vertices = path.vertices
        vertices[:, 0] = np.clip(vertices[:, 0], min(vertices[:, 0]), np.mean(vertices[:, 0]))
        path.vertices = vertices





        
        
old_len_collections = len(vio.collections)

sns.stripplot(data=mae_df, palette=colors, dodge=False, ax=ax, size=10, linewidth=1.5)


# Adjust the position of the stripplot
for dots in vio.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets() + np.array([0.17, 0]))


ax.set_ylabel("CV RSME", fontweight="bold", fontsize=20)
ax.grid(True, linestyle="--")

# ax.legend_.remove()
plt.grid(True, which='major', axis='y')
plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
plt.savefig(os.path.join(path_to_save, "Inter-estimator_rmse.png"), bbox_inches="tight", dpi=300, format="png")