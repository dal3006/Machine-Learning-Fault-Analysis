# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def render_accu_matrix(accu_matrix, datasets):
    ax = sns.heatmap(accu_matrix, annot=True, fmt='.1f', xticklabels=datasets,
                     yticklabels=datasets, cmap='coolwarm', vmin=50, vmax=100)
    ax.set_title("Transfer results")
    ax.set_ylabel("Source")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel("Target")
    return ax


m = np.array([[0.5, 0.7], [0.1, 0.9]])
render_accu_matrix(m, ["aaa", "bbb"])
