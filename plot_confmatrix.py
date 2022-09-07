#%%


from nptdms import TdmsFile
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



MATRICES = [
    {
        'name': "cwru3_a0b0_source",
        'lbls': ["NORM","INNER","OUTER"],
        'values': np.array([[1,0,0],[0,.99,.01],[0,.01,.99]])
    },
    {
        'name': "cwru3_a10b0_source",
        'lbls': ["NORM","INNER","OUTER"],
        'values': np.array([[1,0,0],[0,.99,.01],[0,.02,.98]])
    },
    {
        'name': "cwru3_a0b0_target",
        'lbls': ["NORM","INNER","OUTER"],
        'values': np.array([[.99,.01,0],[0,.86,.14],[0,.01,.99]])
    },
    {
        'name': "cwru3_a10b0_target",
        'lbls': ["NORM","INNER","OUTER"],
        'values': np.array([[1,0,0],[0,.98,.02],[0,.04,.96]])
    },
    {
        'name': "mand_1k30k_notransfer_source",
        'lbls': ["NORM","INNER"],
        'values': np.array([[1,0],[0,1]])
    },
    {
        'name': "mand_1k30k_transfer_source",
        'lbls': ["NORM","INNER"],
        'values': np.array([[1,0],[.01,.99]])
    },
    {
        'name': "mand_1k30k_notransfer_target",
        'lbls': ["NORM","INNER"],
        'values': np.array([[.97,.03],[0.48,0.52]])
    },
    {
        'name': "mand_1k30k_transfer_target",
        'lbls': ["NORM","INNER"],
        'values': np.array([[.96,.04],[.06,.94]])
    },
]

for cm in MATRICES:
    plt.figure(figsize=(5,4.5))
    ax = sns.heatmap(cm['values'], cbar=False, annot=True, fmt='.0%', cmap='coolwarm', xticklabels=cm['lbls'], yticklabels=cm['lbls'])
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 90)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)
    ax.set_xlabel('Predicted value')
    ax.set_ylabel('True value')
    # ax.collections[0].colorbar.ax.tick_params(labelsize=25)
    ax.figure.savefig(f'matrices/{cm["name"]}.pdf')

# %%
