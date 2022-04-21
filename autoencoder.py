# %%
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
import os

# %%

DATASET = "dataset/cwru/0"
CLASSES = ["normal", "B007", "IR007"]
INPUT_LENGTH = 800


def read_class_mat_file(cl_path: str):
    """Read classname.mat and extract data collected by DE sensor"""
    cl_data = loadmat(cl_path)
    # Available sensors are DE, FE, BA. Pick only DE
    key = [k for k in cl_data.keys() if "DE" in k][0]
    de_data = cl_data[key]
    return de_data.flatten()


def split_into_samples(cl_data: np.array, length: int):
    """Given a signal, divide it in n samples of length length"""
    X = []
    for i in range(0, ((len(cl_data) // length) - 1) * length, length):
        X.append(cl_data[i:i + length])
    return np.array(X).reshape((-1, length))


X = []
Y = []
for i, cl in enumerate(CLASSES):
    cl_path = os.path.join(DATASET, cl + ".mat")
    cl_data = read_class_mat_file(cl_path)
    cl_samples = list(split_into_samples(cl_data, INPUT_LENGTH))
    X += cl_samples
    Y += [i] * len(cl_samples)

X = np.array(X)
Y = np.array(Y)
X.shape, Y.shape

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
