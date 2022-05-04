# %%
import glob
from scipy.io import loadmat
import numpy as np
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
import torch

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
    'DE014': {
        'sensor': 'DE',
        'classes': {
            "Normal": "dataset/cwru/*/normal.mat",
            "Ball": "dataset/cwru/*/B014.mat",
            "Inner race": "dataset/cwru/*/IR014.mat",
            "Outer race": "dataset/cwru/*/OR014@*.mat"
        }
    },
    'DE021': {
        'sensor': 'DE',
        'classes': {
            "Normal": "dataset/cwru/*/normal.mat",
            "Ball": "dataset/cwru/*/B021.mat",
            "Inner race": "dataset/cwru/*/IR021.mat",
            "Outer race": "dataset/cwru/*/OR021@*.mat"
        }
    },
    'FE007': {
        'sensor': 'FE',
        'classes': {
            "Normal": "dataset/cwru/*/normal.mat",
            "Ball": "dataset/cwru/*/B007.mat",
            "Inner race": "dataset/cwru/*/IR007.mat",
            "Outer race": "dataset/cwru/*/OR007@*.mat"
        }
    },
    'FE014': {
        'sensor': 'FE',
        'classes': {
            "Normal": "dataset/cwru/*/normal.mat",
            "Ball": "dataset/cwru/*/B014.mat",
            "Inner race": "dataset/cwru/*/IR014.mat",
            "Outer race": "dataset/cwru/*/OR014@*.mat"
        }
    },
    'FE021': {
        'sensor': 'FE',
        'classes': {
            "Normal": "dataset/cwru/*/normal.mat",
            "Ball": "dataset/cwru/*/B021.mat",
            "Inner race": "dataset/cwru/*/IR021.mat",
            "Outer race": "dataset/cwru/*/OR021@*.mat"
        }
    },
    'DE': {
        'sensor': 'DE',
        'classes': {
            "Normal": "dataset/cwru/*/normal.mat",
            "Ball": "dataset/cwru/*/B*.mat",
            "Inner race": "dataset/cwru/*/IR*.mat",
            "Outer race": "dataset/cwru/*/OR*@*.mat"
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
CWRUA = {'sensor': 'DE',
         'classes': {
             "Normal": "dataset/cwru/0/normal.mat",
             "Ball": "dataset/cwru/0/B*.mat",
             "Inner race": "dataset/cwru/0/IR*.mat",
             "Outer race": "dataset/cwru/0/OR*.mat"
         }}

CWRUB = {'sensor': 'DE',
         'classes': {
             "Normal": "dataset/cwru/3/normal*.mat",
             "Ball": "dataset/cwru/3/B*.mat",
             "Inner race": "dataset/cwru/3/IR*.mat",
             "Outer race": "dataset/cwru/3/OR*.mat"
         }}
CLASSES = sorted(CWRUA['classes'].keys())


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


def normalize(x):
    assert x.size(0) > 1
    assert x.size(1) == 1
    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True)
    x = (x - mean) / (std + 1e-12)
    return x


def read_dataset(conf, input_length, train_overlap, test_overlap, test_size):
    """Read dataset from disk and split it into samples"""
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i, (class_name, class_regex) in enumerate(conf['classes'].items()):
        print(f'[{i}] Loading class {class_name}')
        # One class can be split into multiple .mat files, so load them all
        for cl_path in glob.glob(class_regex):
            # Load signal from file
            sig = read_class_mat_file(cl_path, conf['sensor'])
            sig = torch.Tensor(sig)

            if test_size != 0:
                # Handle test set
                sig_train, sig_test = train_test_split(sig, test_size=test_size, random_state=42)
                # Divide continous signal into rolling window samples
                step = int((1 - test_overlap) * input_length)
                sig_test = sig_test.unfold(dimension=0, size=input_length, step=step)
                X_test.append(sig_test)
                Y_test.append(torch.Tensor([i] * sig_test.size(0)))
            else:
                sig_train = sig

            # Handle train set
            step = int((1 - train_overlap) * input_length)
            sig_train = sig_train.unfold(dimension=0, size=input_length, step=step)
            # Append to dataset with labels
            X_train.append(sig_train)
            Y_train.append(torch.Tensor([i] * sig_train.size(0)))

    X_train = normalize(torch.cat(X_train).unsqueeze(1))
    Y_train = torch.cat(Y_train).type(torch.LongTensor)
    if test_size != 0:
        X_test = normalize(torch.cat(X_test).unsqueeze(1))
        Y_test = torch.cat(Y_test).type(torch.LongTensor)
    return X_train, Y_train, X_test, Y_test


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
