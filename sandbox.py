# %%
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import tensorboard as tb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

labels = ["NORM", "INNER", "OUTER", "BALL"]
cm = np.random.normal(0, 1, (4, 4))

# replot the heatmap with y labels rotated
plt.figure(figsize=(18, 13))
g = sns.heatmap(cm, annot=True, fmt='.2f', cmap='coolwarm', xticklabels=labels, yticklabels=labels, annot_kws={'size': 25})
g.set_yticklabels(g.get_yticklabels(), rotation = 90, fontsize=25)
g.set_xticklabels(g.get_xticklabels(), rotation = 0, fontsize=25)
g.set_xlabel('Predicted class', fontsize=25, labelpad=35)
g.set_ylabel('True class', fontsize=25, labelpad=35)
g.collections[0].colorbar.ax.tick_params(labelsize=25)
# g.collections[0].colorbar.ax.yaxis.labelpad = 50
# g.collections[0].colorbar.ax.xaxis.labelpad = 50
plt.show()

# %%
