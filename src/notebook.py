# %%
!"$(pip install --upgrade pip)"
!"$(pip install pandas scikit-learn numpy matplotlib seaborn)"
# %%
%load_ext autoreload
%autoreload 2
# %%
from src.eval import CWRUA, CWRUB, read_dataset
from src.model import create_model, compile_model, train_model, INPUT_LENGTH
from scipy.io import loadmat
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, losses, utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
X_train, y_train = read_dataset(CWRUA, input_length=INPUT_LENGTH)
X_test, y_test = read_dataset(CWRUB, input_length=INPUT_LENGTH)

# %%
model = create_model(INPUT_LENGTH)
model.summary()

# %%
model = compile_model(model)
model, history = train_model(model, X_train, y_train, validation_data=(X_test, y_test))

# %%
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
# %%


def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    # print("Precision = {}".format(precision_score(labels, predictions)))
    # print("Recall = {}".format(recall_score(labels, predictions)))


y_hat = model(X_test)
print_stats(np.argmax(y_hat, axis=1), np.argmax(y_test, axis=1))

# %%
