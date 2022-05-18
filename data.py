from typing import List
from pytorch_lightning.core import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import glob
from scipy.io import loadmat
import numpy as np
import os

# Note that 'classes' order matter
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
              }},
    'CWRUAfd': {'sensor': 'DE',
                'classes': {
                    "Normal": "cwru/0/normal.mat",
                    "Fault": "cwru/0/*0*.mat",
                }},
    'CWRUBfd': {'sensor': 'DE',
                'classes': {
                    "Normal": "cwru/3/normal*.mat",
                    "Fault": "cwru/3/*0*.mat",
                }},
    'CWRUA3': {'sensor': 'DE',
              'classes': {
                  "Normal": "cwru/0/normal.mat",
                  "Inner race": "cwru/0/IR*.mat",
                  "Outer race": "cwru/0/OR*.mat"
              }},
    'CWRUB3': {'sensor': 'DE',
              'classes': {
                  "Normal": "cwru/3/normal*.mat",
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
                 reuse_target: bool
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
        self.reuse_target = reuse_target

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("MyDataModule")
        parser.add_argument("--data_dir", type=str, default="./dataset/")
        parser.add_argument("--source", type=str, default="CWRUA")
        parser.add_argument("--target", type=str, default="CWRUB")
        parser.add_argument("--test_size", type=float, default=0.2)
        parser.add_argument("--input_length", type=int, default=256)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--reuse_target", default="false",
                            type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
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
            y_trg_train = y_trg_train[0:src_sz]


        print("y_src_train")
        classes, counts = y_src_train.unique(return_counts=True)
        ds_size = y_src_train.size(0)
        percents = counts / ds_size * 100
        print("CLASS\tCOUNT\tPERC\tWEIGHT")
        for cl, cnt, perc in zip(classes, counts, percents):
            print(f'{int(cl)}\t{cnt}\t{perc:.1f}%')

        print("y_trg_train")
        classes, counts = y_trg_train.unique(return_counts=True)
        ds_size = y_trg_train.size(0)
        percents = counts / ds_size * 100
        print("CLASS\tCOUNT\tPERC\tWEIGHT")
        for cl, cnt, perc in zip(classes, counts, percents):
            print(f'{int(cl)}\t{cnt}\t{perc:.1f}%')

        print("y_src_test")
        classes, counts = y_src_test.unique(return_counts=True)
        ds_size = y_src_test.size(0)
        percents = counts / ds_size * 100
        print("CLASS\tCOUNT\tPERC\tWEIGHT")
        for cl, cnt, perc in zip(classes, counts, percents):
            print(f'{int(cl)}\t{cnt}\t{perc:.1f}%')

        print("y_trg_test")
        classes, counts = y_trg_test.unique(return_counts=True)
        ds_size = y_trg_test.size(0)
        percents = counts / ds_size * 100
        print("CLASS\tCOUNT\tPERC\tWEIGHT")
        for cl, cnt, perc in zip(classes, counts, percents):
            print(f'{int(cl)}\t{cnt}\t{perc:.1f}%')


        # Train
        self.x_src_train = x_src_train
        self.x_trg_train = x_trg_train
        self.y_src_train = y_src_train
        self.y_trg_train = y_trg_train
        # Test
        self.x_src_test = x_src_test
        self.x_trg_test = x_trg_test
        self.y_src_test = y_src_test
        self.y_trg_test = y_trg_test

    def train_dataloader(self):
        dataset = TensorDataset(self.x_src_train, self.x_trg_train, self.y_src_train)
        train_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size, num_workers=8)
        return train_loader

    def val_dataloader(self):
        dataset = TensorDataset(self.x_src_test, self.y_src_test)
        src_test_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8)
        if self.reuse_target:
            print("WARNING: target reuse is enabled")
            dataset = TensorDataset(self.x_trg_train, self.y_trg_train)
        else:
            dataset = TensorDataset(self.x_trg_test, self.y_trg_test)
        trg_test_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8)

        return [src_test_loader, trg_test_loader]


class AddGaussianNoise(object):
    def __init__(self, sigma):
        assert isinstance(sigma, (float, tuple))
        self.sigma = sigma

    def __call__(self, sample):
        noise = torch.normal(0, 0.25, sample['data'].size())
        return {'data': sample['data'] + noise, 'label': sample['label']}


def plot_batch(x_batch, label: str):
    """
    x_batch: (batch_sz, input_length)
    label: 1d array
    """
    fig, axs = plt.subplots(len(x_batch))
    plt.title(label)
    for x, ax in zip(x_batch, axs):
        ax.plot(x)
    plt.show()


def plot_batch_multilabel(x, y):
    """
    x_batch: (batch_sz, input_length)
    label: (batch_sz, 1)
    """
    batch_sz = len(x)
    _, counts = torch.unique(y, return_counts=True)
    curr_row_idxs = [0 for x in range(len(counts))]
    plt.figure()
    fig, axs = plt.subplots(int(max(counts)), len(counts))
    for x, y in zip(x, y):
        col_idx = int(y)
        row_idx = curr_row_idxs[col_idx]
        curr_row_idxs[col_idx] += 1
        axs[row_idx, col_idx].plot(x)
        axs[0, col_idx].set_title(col_idx)
    plt.show()


def read_dataset(root_dir, conf, input_length, train_overlap, test_overlap, test_size):
    """Read dataset from disk and split it into samples"""
    x_train = []
    x_test = []
    for i, (class_name, class_regex) in enumerate(conf['classes'].items()):
        print(f'[{i}] Loading class {class_name}')

        # One class can be split into multiple .mat files, so load them all
        class_sampl_train = []
        class_sampl_test = []
        for cl_path in glob.glob(os.path.join(root_dir, class_regex)):
            print(cl_path)
            # Load signal from file
            sig = read_class_mat_file(cl_path, conf['sensor'])
            sig = torch.tensor(sig)

            # plot_batch(sig[:-(len(sig) % 256)].view(-1, 256)[:32], cl_path)

            # Split into train/test
            split_idx = int(sig.size(0) * (1 - test_size))
            # Train
            sig_train = sig[:split_idx]
            step = int((1 - train_overlap) * input_length)
            sig_train_samples = sig_train.unfold(dimension=0, size=input_length, step=step)
            class_sampl_train.append(sig_train_samples)
            # Test
            sig_test = sig[split_idx:]
            step = int((1 - test_overlap) * input_length)
            sig_test_samples = sig_test.unfold(dimension=0, size=input_length, step=step)
            class_sampl_test.append(sig_test_samples)

            # plt.figure(figsize=(16, 2))
            # plt.plot(class_sampl_test[-1][0])
            # plt.title(cl_path)
            # plt.show()

        # merge class samples from different files
        x_train.append(torch.cat(class_sampl_train))
        x_test.append(torch.cat(class_sampl_test))

    # Rebalance
    x_train = rebalance_by_removal(x_train)
    x_test = rebalance_by_removal(x_test)
    # Create labels
    y_train = [[i] * len(x) for i, x in enumerate(x_train)]
    y_test = [[i] * len(x) for i, x in enumerate(x_test)]
    # Reshape
    x_train = torch.cat(x_train).type(torch.FloatTensor)
    x_test = torch.cat(x_test).type(torch.FloatTensor)
    y_train = torch.tensor(y_train).reshape(-1).type(torch.LongTensor)
    y_test = torch.tensor(y_test).reshape(-1).type(torch.LongTensor)
    # Normalize
    x_train = minmax_normalization(x_train)
    x_test = minmax_normalization(x_test)

    return x_train.unsqueeze(1), y_train, x_test.unsqueeze(1), y_test


def rebalance_by_removal(classes_samples: List):
    """classes_samples is a list of torch tensors: [[class1],[class2],....]"""
    # Rebalance classes
    num_classes = len(classes_samples)
    lengths = [len(cl) for cl in classes_samples]
    min_length = min(lengths)
    print("Lengths before are: " + str(lengths))
    for i in range(num_classes):
        # random choice without repetition
        indices = torch.randperm(lengths[i])[:min_length]
        classes_samples[i] = classes_samples[i][indices]
    new_lengths = [len(cl) for cl in classes_samples]
    print("Lengths after are:  " + str(new_lengths))
    return classes_samples


def std_normalization(x):
    assert len(x.shape) == 2
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    x = (x - mean) / (std + 1e-12)
    return x

def minmax_normalization(x):
    min_v = x.min(axis=1, keepdims=True)[0]
    range_v = x.max(axis=1, keepdims=True)[0] - min_v
    x = (x - min_v) / range_v
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


if __name__ == "__main__":
    """Visualize some batches of data"""
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    # PARSE ARGS
    parser = ArgumentParser()
    parser = MyDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    data_module = MyDataModule.from_argparse_args(args)

    data_module.prepare_data()
    dataloader = data_module.train_dataloader()
    batch = next(iter(dataloader))

    x_s, x_t, y_s = batch
    plot_batch_multilabel(x_s.squeeze(1), y_s)
