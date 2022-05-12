# %%
import numpy as np
from data import read_dataset, DATASETS
import torch
# %%
a = np.load("FRAN/CWRU_dataset/CWRU_DE.npy", allow_pickle=True)


x_src_train, y_src_train, x_src_test, y_src_test = read_dataset("dataset",
                                                                DATASETS["CWRUA"],
                                                                test_size=0.2,
                                                                input_length=256,
                                                                train_overlap=0.8,
                                                                test_overlap=0.8)


classes, counts = y_src_train.unique(return_counts=True)
print(classes)
print(counts)
min_idx = torch.argmin(counts)
min_count = int(counts[min_idx])
min_idx, min_count

for cl in classes:
    y_src_train = y_src_train.numpy()
    choices = np.random.choice(y_src_train[y_src_train == cl], min_count)

# %%


x = torch.randint(1, 10, (4, 2))
print(x)
# %%
indices = torch.randperm(len(x))[:4]
x[indices]
