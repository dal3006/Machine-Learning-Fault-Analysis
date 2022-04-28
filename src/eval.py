import glob
from pdb import post_mortem
from scipy.io import loadmat
import numpy as np
from tensorflow.keras import utils
import seaborn as sns
import matplotlib.pyplot as plt

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


def read_dataset(conf, input_length):
    """Read dataset from disk and split it into samples"""
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
            y_hat = model(test_x)
            accu = accuracy_score(np.argmax(test_y, axis=1), np.argmax(y_hat, axis=1))
            accu_matrix[row, col] = accu

    # Display results
    print(accu_matrix)
    ax = render_accu_matrix(accu_matrix*100, DATASETS.keys())
    ax.figure.savefig("out.svg")
    ax.figure.savefig("out.png")


if __name__ == '__main__':
    main()
