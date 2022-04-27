# %%
import os
import glob
from scipy.io import loadmat
import numpy as np
from tensorflow.keras import layers, losses, utils

# %%
BASE_PATH = "dataset/cwru"
DATASETS = {
    'DE007': {
        'sensor': 'DE',
        'classes': {
            "Normal": "dataset/cwru/*/normal.mat",
            "Ball": "dataset/cwru/*/B007.mat",
            "Inner race": "dataset/cwru/*/IR007.mat",
            "Outer race": "dataset/cwru/*/OR007@*.mat"
        }
    },
    'FE': {
        'sensor': 'FE',
        'classes': {
            "Normal": "dataset/cwru/*/normal.mat",
            "Ball": "dataset/cwru/*/B*.mat",
            "Inner race": "dataset/cwru/*/IR*.mat",
            "Outer race": "dataset/cwru/*/OR*@*.mat"
        }
    }
}


def read_class_mat_file(cl_files_regx: str, sensor: str):
    """Read classname.mat and extract data collected by specified sensor"""
    sensor_data = []
    for cl_path in glob.glob(cl_files_regx):
        cl_data = loadmat(cl_path)
        # Available sensors are DE, FE, BA. Note that in some measurements
        # not all sensors are available
        keys = [k for k in cl_data.keys() if sensor in k]
        if len(keys) > 0:
            sensor_data += list(cl_data[keys[0]].flatten())
        else:
            print(f'Warning: sensor {sensor} is missing in {cl_path}')
    return sensor_data


def split_into_samples(cl_data: np.array, length: int):
    """Given a signal, divide it in n samples of length length"""
    X = []
    for i in range(0, ((len(cl_data) // length) - 1) * length, length):
        X.append(cl_data[i:i + length])
    return np.array(X).reshape((-1, length))


def read_dataset(conf, input_length):
    X = []
    Y = []
    for i, (class_name, class_regex) in enumerate(conf['classes'].items()):
        print(f'[{i}] Loading class {class_name}')
        # One class can be split into multiple .mat files, so load them all
        cl_samples = []
        for cl_path in glob.glob(class_regex):
            print(f'{cl_path}')
            cl_data = read_class_mat_file(cl_path, conf['sensor'])
            cl_samples += list(split_into_samples(cl_data, input_length))
        X += cl_samples
        Y += [i] * len(cl_samples)

    X = np.array(X)
    Y = np.array(utils.to_categorical(Y))
    return X, Y


INPUT_LENGTH = 500

for train_name, train_conf in DATASETS.items():
    train_x, train_y = read_dataset(train_conf, INPUT_LENGTH)
    for test_name, test_conf in DATASETS.items():
        test_x, test_y = read_dataset(test_conf, INPUT_LENGTH)
        print(f'Train on {train_name} and eval on {test_name}')
