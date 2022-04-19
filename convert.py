# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

data_dict = loadmat('dataset/normal_0.mat')
x = data_dict["X097_DE_time"]


plt.figure()
plt.plot(x)
plt.show()


# %%
X = []
N = 1000
for i in range(0, ((len(x) // N) - 1)*N, N):
    X.append(x[i:i+N])

X = np.array(X).reshape((-1,N))
X.shape

plt.figure()
plt.plot(X[0])
plt.show()

# %%
