

from pytorch_lightning.core import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import glob
from scipy.io import loadmat
import numpy as np
import os

DATASETS = {
    'DE007': {
        'sensor': 'DE',
        'classes': {
            "Normal": "cwru/*/normal.mat",
            "Ball": "cwru/*/B007.mat",
            "Inner race": "cwru/*/IR007.mat",
            "Outer race": "cwru/*/OR007@*.mat"
        }
    },
    'DE014': {
        'sensor': 'DE',
        'classes': {
            "Normal": "cwru/*/normal.mat",
            "Ball": "cwru/*/B014.mat",
            "Inner race": "cwru/*/IR014.mat",
            "Outer race": "cwru/*/OR014@*.mat"
        }
    },
    'DE021': {
        'sensor': 'DE',
        'classes': {
            "Normal": "cwru/*/normal.mat",
            "Ball": "cwru/*/B021.mat",
            "Inner race": "cwru/*/IR021.mat",
            "Outer race": "cwru/*/OR021@*.mat"
        }
    },
    'FE007': {
        'sensor': 'FE',
        'classes': {
            "Normal": "cwru/*/normal.mat",
            "Ball": "cwru/*/B007.mat",
            "Inner race": "cwru/*/IR007.mat",
            "Outer race": "cwru/*/OR007@*.mat"
        }
    },
    'FE014': {
        'sensor': 'FE',
        'classes': {
            "Normal": "cwru/*/normal.mat",
            "Ball": "cwru/*/B014.mat",
            "Inner race": "cwru/*/IR014.mat",
            "Outer race": "cwru/*/OR014@*.mat"
        }
    },
    'FE021': {
        'sensor': 'FE',
        'classes': {
            "Normal": "cwru/*/normal.mat",
            "Ball": "cwru/*/B021.mat",
            "Inner race": "cwru/*/IR021.mat",
            "Outer race": "cwru/*/OR021@*.mat"
        }
    },
    'DE': {
        'sensor': 'DE',
        'classes': {
            "Normal": "cwru/*/normal.mat",
            "Ball": "cwru/*/B*.mat",
            "Inner race": "cwru/*/IR*.mat",
            "Outer race": "cwru/*/OR*@*.mat"
        }
    },
    'FE': {
        'sensor': 'FE',
        'classes': {
            "Normal": "cwru/*/normal.mat",
            "Ball": "cwru/*/B*.mat",
            "Inner race": "cwru/*/IR*.mat",
            "Outer race": "cwru/*/OR*@*.mat"
        }
    },
    'CWRUA': {'sensor': 'DE',
              'classes': {
                  "Normal": "cwru/0/normal.mat",
                  "Ball": "cwru/0/B*.mat",
                  "Inner race": "cwru/0/IR*.mat",
                  "Outer race": "cwru/0/OR*.mat"
              }},
    'CWRUB': {'sensor': 'DE',
              'classes': {
                  "Normal": "cwru/3/normal*.mat",
                  "Ball": "cwru/3/B*.mat",
                  "Inner race": "cwru/3/IR*.mat",
                  "Outer race": "cwru/3/OR*.mat"
              }}
}


class MyDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 source: str,
                 target: str,
                 test_size: float,
                 input_length: int,
                 batch_size: int,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.source = source
        self.target = target
        self.source_conf = DATASETS[source]
        self.target_conf = DATASETS[target]
        self.test_size = test_size
        self.input_length = input_length
        self.batch_size = batch_size
        self.class_weights = None

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("MyDataModule")
        parser.add_argument("--data_dir", type=str, default="./dataset/")
        parser.add_argument("--source", type=str, default="CWRUA")
        parser.add_argument("--target", type=str, default="CWRUB")
        parser.add_argument("--test_size", type=float, default=0.1)
        parser.add_argument("--input_length", type=int, default=256)
        parser.add_argument("--batch_size", type=int, default=256)
        return parent_parser

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        # Load data
        x_src_train, y_src_train, x_src_test, y_src_test = read_dataset(self.data_dir,
                                                                        self.source_conf,
                                                                        test_size=self.test_size,
                                                                        input_length=self.input_length,
                                                                        train_overlap=0.8,
                                                                        test_overlap=0.8)
        x_trg_train, y_trg_train, x_trg_test, y_trg_test = read_dataset(self.data_dir,
                                                                        self.target_conf,
                                                                        test_size=self.test_size,
                                                                        input_length=self.input_length,
                                                                        train_overlap=0.8,
                                                                        test_overlap=0.8)
        # Cut to the same length
        src_sz = x_src_train.size(0)
        trg_sz = x_trg_train.size(0)
        if src_sz > trg_sz:
            # Source bigger
            x_src_train = x_src_train[0:trg_sz]
            y_src_train = y_src_train[0:trg_sz]
        elif trg_sz > src_sz:
            # Target bigger
            x_trg_train = x_trg_train[0:src_sz]

        classes, counts = y_src_train.unique(return_counts=True)
        ds_size = y_src_train.size(0)
        percents = counts / ds_size * 100
        class_weights = ds_size / (len(classes) * counts)
        print("CLASS\tCOUNT\tPERC\tWEIGHT")
        for cl, cnt, perc, wght in zip(classes, counts, percents, class_weights):
            print(f'{cl}\t{cnt}\t{perc:.1f}%\t{wght:.3f}')

        # Train
        self.class_weights = class_weights
        self.x_src_train = x_src_train
        self.x_trg_train = x_trg_train
        self.y_src_train = y_src_train
        # Test
        self.x_src_test = x_src_test
        self.y_src_test = y_src_test
        self.x_trg_test = x_trg_test
        self.y_trg_test = y_trg_test

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        dataset = TensorDataset(self.x_src_train, self.x_trg_train, self.y_src_train)
        train_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size, num_workers=8)
        return train_loader

    def val_dataloader(self):
        dataset = TensorDataset(self.x_src_test, self.y_src_test)
        src_test_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8)
        dataset = TensorDataset(self.x_trg_test, self.y_trg_test)
        trg_test_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8)

        return [src_test_loader, trg_test_loader]

    # def test_dataloader(self):
    #     test_split = Dataset(...)
    #     return DataLoader(test_split)

    # def teardown(self):
        # clean up after fit or test
        # called on every process in DDP


class AddGaussianNoise(object):
    def __init__(self, sigma):
        assert isinstance(sigma, (float, tuple))
        self.sigma = sigma

    def __call__(self, sample):
        noise = torch.normal(0, 0.25, sample['data'].size())
        return {'data': sample['data'] + noise, 'label': sample['label']}


def read_dataset(root_dir, conf, input_length, train_overlap, test_overlap, test_size):
    """Read dataset from disk and split it into samples"""
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i, (class_name, class_regex) in enumerate(conf['classes'].items()):
        print(f'[{i}] Loading class {class_name}')
        # One class can be split into multiple .mat files, so load them all

        for cl_path in glob.glob(os.path.join(root_dir, class_regex)):
            # Load signal from file
            sig = read_class_mat_file(cl_path, conf['sensor'])
            sig = torch.Tensor(sig)

            # plt.figure(figsize=(16, 2))
            # plt.plot(sig[0:input_length])
            # plt.title(cl_path)
            # plt.show()

            # Split into train/test
            split_idx = int(sig.size(0) * (1 - test_size))
            # Train
            sig_train = sig[:split_idx]
            step = int((1 - train_overlap) * input_length)
            sig_train = sig_train.unfold(dimension=0, size=input_length, step=step)
            X_train.append(sig_train)
            Y_train.append(torch.Tensor([i] * sig_train.size(0)))
            # Test
            sig_test = sig[split_idx:]
            step = int((1 - test_overlap) * input_length)
            sig_test = sig_test.unfold(dimension=0, size=input_length, step=step)
            X_test.append(sig_test)
            Y_test.append(torch.Tensor([i] * sig_test.size(0)))

            # plt.figure(figsize=(16, 2))
            # plt.plot(X_train[-1][0])
            # plt.title(cl_path)
            # plt.show()

    X_train = normalize(torch.cat(X_train).unsqueeze(1))
    Y_train = torch.cat(Y_train).type(torch.LongTensor)
    X_test = normalize(torch.cat(X_test).unsqueeze(1))
    Y_test = torch.cat(Y_test).type(torch.LongTensor)
    return X_train, Y_train, X_test, Y_test


def normalize(x):
    assert x.size(0) > 1
    assert x.size(1) == 1
    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True)
    x = (x - mean) / (std + 1e-12)
    return x


def read_class_mat_file(cl_path: str, sensor: str):
    """Read classname.mat and extract data collected by specified sensor"""
    cl_data = loadmat(cl_path)
    # Available sensors are DE, FE, BA. Note that in some measurements
    # not all sensors are available
    keys = [k for k in cl_data.keys() if sensor in k]
    if len(keys) > 0:
        sig = cl_data[keys[0]].flatten()
    else:
        print(f'Warning: sensor {sensor} is missing in {cl_path}')
        sig = np.array()
    return sig
