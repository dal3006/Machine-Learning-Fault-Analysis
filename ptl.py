# %%
# %load_ext autoreload
# %autoreload 2

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data import CWRUA, CWRUB, read_dataset
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torchmetrics

INPUT_LENGTH = 256
BATCH_SZ = 256
TRAIN_OVERLAP = 0.8
TEST_OVERLAP = 0.8
TEST_SIZE = 0.1
ENABLE_MMD = True
ALPHA = 1e-4
LEARNING_RATE = 1e-03
# %%





dataset = TensorDataset(x_src_train, x_trg_train, y_src_train)
train_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SZ, num_workers=8)
dataset = TensorDataset(x_src_test, y_src_test)
src_test_loader = DataLoader(dataset, batch_size=BATCH_SZ, num_workers=8)
dataset = TensorDataset(x_trg_test, y_trg_test)
trg_test_loader = DataLoader(dataset, batch_size=BATCH_SZ, num_workers=8)

# %%



# %%
# [x1, x2, y1] = next(iter(train_loader))
# for x_s, x_t, y_s in zip(x1, x2, y1):
#     if int(y_s) == 0:
#         plt.figure(figsize=(16, 2))
#         plt.title(y_s)
#         # for i in range(10):
#         #     noised = AddGaussianNoise(0.25)({'data': x_s, 'label': y_s})
#         #     plt.plot(noised['data'].squeeze(0))
#         plt.plot(x_s.squeeze(0))
#         plt.show()

# %%




# %%
