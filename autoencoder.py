# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, utils
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import os
import glob

# %%

CWRUA = {
    "Normal": "dataset/cwru/0/normal*.mat",
    "Ball": "dataset/cwru/0/B*.mat",
    "Inner race": "dataset/cwru/0/IR*.mat",
    "Outer race": "dataset/cwru/0/OR*.mat"
}
CWRUB = {
    "Normal": "dataset/cwru/3/normal*.mat",
    "Ball": "dataset/cwru/3/B*.mat",
    "Inner race": "dataset/cwru/3/IR*.mat",
    "Outer race": "dataset/cwru/3/OR*.mat"
}
CLASSES = sorted(CWRUA.keys())
INPUT_LENGTH = 800
# %%


def read_class_mat_file(cl_files_regx: str):
    """Read classname.mat and extract data collected by DE sensor"""
    sensor_data = []
    for cl_path in glob.glob(cl_files_regx):
        cl_data = loadmat(cl_path)
        # Available sensors are DE, FE, BA. Pick only DE
        key = [k for k in cl_data.keys() if "DE" in k][0]
        sensor_data += list(cl_data[key].flatten())
    return sensor_data


def split_into_samples(cl_data: np.array, length: int):
    """Given a signal, divide it in n samples of length length"""
    X = []
    for i in range(0, ((len(cl_data) // length) - 1) * length, length):
        X.append(cl_data[i:i + length])
    return np.array(X).reshape((-1, length))


def read_dataset(class_files):
    X = []
    Y = []
    for i, cl in enumerate(CLASSES):
        print(f'[{i}] Loading class {cl}')
        # One class can be split into multiple .mat files, so load them all
        cl_samples = []
        for cl_path in glob.glob(class_files[cl]):
            print(f'{cl_path}')
            cl_data = read_class_mat_file(cl_path)
            cl_samples += list(split_into_samples(cl_data, INPUT_LENGTH))
        X += cl_samples
        Y += [i] * len(cl_samples)

    X = np.array(X)
    Y = np.array(utils.to_categorical(Y))
    return X, Y


X_train, y_train = read_dataset(CWRUA)
X_test, y_test = read_dataset(CWRUB)


# %%
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# print(f'{X_train.shape=} {X_test.shape=}')
# print(f'{y_train.shape=} {y_test.shape=}')

# %%


def model(input_length):
    x = input_tensor = tf.keras.Input(shape=(input_length, 1), name="raw_signal")

    # Encoder
    x = layers.Conv1D(filters=32, kernel_size=2, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(filters=32, kernel_size=2, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(filters=32, kernel_size=2, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    embeddings = layers.Flatten()(x)

    # x = layers.UpSampling1D(2)(x)
    # x = layers.Conv1DTranspose(filters=32, kernel_size=2, padding="same")(x)
    # x = layers.UpSampling1D(2)(x)
    # x = layers.Conv1DTranspose(filters=1, kernel_size=2, padding="same")(x)

    x = layers.Dense(10)(embeddings)
    x = layers.Dense(4, activation="softmax")(x)

    output = x
    return tf.keras.Model(
        inputs=[input_tensor],
        outputs=output,
    )


m = model(INPUT_LENGTH)
m.summary()


# %%

m.compile(optimizer='adam', loss='categorical_crossentropy')

history = m.fit(X_train, y_train,
                epochs=50,
                batch_size=512,
                validation_data=(X_test, y_test),
                shuffle=True)

# %%
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

# %%


def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    # print("Precision = {}".format(precision_score(labels, predictions)))
    # print("Recall = {}".format(recall_score(labels, predictions)))


y_hat = m(X_test)
print_stats(np.argmax(y_hat, axis=1), np.argmax(y_test, axis=1))
