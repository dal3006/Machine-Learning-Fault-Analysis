# %%
from tkinter import _Padding
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
# %%
data_dict = loadmat('dataset/normal_0.mat')
x = data_dict["X097_DE_time"]

X = []
N = 600
for i in range(0, ((len(x) // N) - 1) * N, N):
    X.append(x[i:i + N])

X = np.array(X).reshape((-1, N))
print(X.shape)

plt.figure()
plt.plot(X[0])
plt.show()

# %%
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
print(f'{X_train.shape=} {X_test.shape=}')

# %%


def model():
    x = input_tensor = tf.keras.Input(shape=(N, 1), name="raw_signal")

    # Encoder
    x = layers.Conv1D(filters=32, kernel_size=2, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(filters=64, kernel_size=2, padding="same", activation="relu")(x)
    embeddings = x = layers.MaxPool1D(2)(x)

    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1DTranspose(filters=32, kernel_size=2, padding="same")(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1DTranspose(filters=1, kernel_size=2, padding="same")(x)
    output = x
    return tf.keras.Model(
        inputs=[input_tensor],
        outputs=output,
    )


autoencoder = model()
autoencoder.summary()


# %%

autoencoder.compile(optimizer='adam', loss='mae')

history = autoencoder.fit(X_train, X_train,
                          epochs=200,
                          batch_size=512,
                          validation_data=(X_test, X_test),
                          shuffle=True)

# %%
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

# %%
