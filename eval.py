# %%
import glob
from scipy.io import loadmat
import numpy as np
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt

BASE_PATH = "dataset/cwru"


def render_accu_matrix(accu_matrix, datasets):
    ax = sns.heatmap(accu_matrix, annot=True, fmt='.1f', xticklabels=datasets,
                     yticklabels=datasets, cmap='coolwarm', vmin=50, vmax=100)
    ax.set_title("Fault Severity Diagnosis")
    ax.set_ylabel("Source")
    ax.set_xlabel("Target")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    return ax


def main():
    """Main loop"""
    from model import create_model, compile_model, train_model, INPUT_LENGTH
    from sklearn.metrics import accuracy_score

    n = len(DATASETS.keys())
    accu_matrix = np.zeros((n, n))

    for row, (train_name, train_conf) in enumerate(DATASETS.items()):
        train_x, train_y = read_dataset(train_conf, INPUT_LENGTH)
        model = create_model(INPUT_LENGTH)
        model = compile_model(model)
        model, _ = train_model(model, train_x, train_y)

        for col, (test_name, test_conf) in enumerate(DATASETS.items()):
            print(f'Train on {train_name} and eval on {test_name}')
            test_x, test_y = read_dataset(test_conf, INPUT_LENGTH)
            y_hat = model.predict(test_x)
            accu = accuracy_score(np.argmax(test_y, axis=1), np.argmax(y_hat, axis=1))
            accu_matrix[row, col] = accu

    # Display results
    print(accu_matrix)
    ax = render_accu_matrix(accu_matrix * 100, DATASETS.keys())
    ax.figure.savefig("out.svg")
    ax.figure.savefig("out.png")


if __name__ == '__main__':
    main()
