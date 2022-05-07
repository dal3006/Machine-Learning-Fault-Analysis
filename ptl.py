# %%
# %load_ext autoreload
# %autoreload 2

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from eval import CWRUA, CWRUB, read_dataset
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torchmetrics

INPUT_LENGTH = 256
BATCH_SZ = 256
TRAIN_OVERLAP = 0.8
TEST_OVERLAP = 0.8
TEST_SIZE = 0.1
ENABLE_MMD = True
ALPHA = 1e-4
LEARNING_RATE = 1e-03
# %%
x_src_train, y_src_train, x_src_test, y_src_test = read_dataset(CWRUA,
                                                                test_size=TEST_SIZE,
                                                                input_length=INPUT_LENGTH,
                                                                train_overlap=TRAIN_OVERLAP,
                                                                test_overlap=TEST_OVERLAP)
x_trg_train, y_trg_train, x_trg_test, y_trg_test = read_dataset(CWRUB,
                                                                test_size=TEST_SIZE,
                                                                input_length=INPUT_LENGTH,
                                                                train_overlap=TRAIN_OVERLAP,
                                                                test_overlap=TEST_OVERLAP)

# %%
# for x_s, x_t, y_s in zip(x_src_train, x_src_test, y_src_train):
#     plt.figure(figsize=(16, 2))
#     plt.plot(x_s.squeeze(0))
#     plt.show()
#     print(y_s)

# %%
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
# class_weights = torch.Tensor([0, 1, 0, 0])  # TODO:zzzz
print("CLASS\tCOUNT\tPERC\tWEIGHT")
for cl, cnt, perc, wght in zip(classes, counts, percents, class_weights):
    print(f'{cl}\t{cnt}\t{perc:.1f}%\t{wght:.3f}')


dataset = TensorDataset(x_src_train, x_trg_train, y_src_train)
train_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SZ, num_workers=8)
dataset = TensorDataset(x_src_test, y_src_test)
src_test_loader = DataLoader(dataset, batch_size=BATCH_SZ, num_workers=8)
dataset = TensorDataset(x_trg_test, y_trg_test)
trg_test_loader = DataLoader(dataset, batch_size=BATCH_SZ, num_workers=8)

# %%


class AddGaussianNoise(object):
    def __init__(self, sigma):
        assert isinstance(sigma, (float, tuple))
        self.sigma = sigma

    def __call__(self, sample):
        noise = torch.normal(0, 0.25, sample['data'].size())
        return {'data': sample['data'] + noise, 'label': sample['label']}


# %%
# [x1, x2, y1] = next(iter(train_loader))
# for x_s, x_t, y_s in zip(x1, x2, y1):
#     if int(y_s) == 0:
#         plt.figure(figsize=(16, 2))
#         plt.title(y_s)
#         # for i in range(10):
#         #     noised = AddGaussianNoise(0.25)({'data': x_s, 'label': y_s})
#         #     plt.plot(noised['data'].squeeze(0))
#         plt.plot(x_s.squeeze(0))
#         plt.show()

# %%


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul,
                                       kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class MyModel(pl.LightningModule):
    def __init__(self, learning_rate, enable_mmd, alpha, class_weights=None,
                 # Fake params, for hparams logging
                 input_length=INPUT_LENGTH,
                 batch_sz=BATCH_SZ,
                 train_overlap=TRAIN_OVERLAP,
                 test_overlap=TEST_OVERLAP,
                 test_size=TEST_SIZE,):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(batch_sz, 1, input_length)
        self.metrics = torch.nn.ModuleDict({
            'cm': torchmetrics.ConfusionMatrix(num_classes=4, normalize="true"),
            'accuracy': torchmetrics.Accuracy()
        })
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        if self.hparams.enable_mmd:
            self.mmd = MMD_loss()

        self.classifier = nn.Sequential(
            nn.Linear(448, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )
        self.softmax = nn.Softmax(dim=1)
        self.crossentropy_loss = nn.CrossEntropyLoss(weight=self.hparams.class_weights)

    def forward(self, x):
        """in lightning, forward defines the prediction/inference actions"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        y_hat = self.softmax(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x_src, x_trg, y_src = batch
        # Extract features
        x_src = self.encoder(x_src)
        x_src = x_src.view(x_src.size(0), -1)  # flatten all dimensions except batch

        if self.hparams.enable_mmd:
            x_trg = self.encoder(x_trg)
            x_trg = x_trg.view(x_trg.size(0), -1)
            mmd_loss = self.mmd(x_src, x_trg) * self.hparams.alpha
        else:
            mmd_loss = 0.0

        # Classify
        x_src = self.classifier(x_src)
        classif_loss = self.crossentropy_loss(x_src, y_src)

        total_loss = classif_loss + mmd_loss
        self.log("classificaiton_loss/train", classif_loss)
        self.log("mmd_loss/train", mmd_loss)
        self.log("total_loss/train", total_loss)
        self.log("hp_metric", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        loss = self.crossentropy_loss(x, y)
        y_hat = self.softmax(x)
        self.log("classificaiton_loss/val", loss, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'preds': torch.argmax(y_hat, axis=1), 'target': y, 'dataloader_idx': dataloader_idx}

    def validation_epoch_end(self, dataloaders_outputs):
        for outputs in dataloaders_outputs:
            preds = torch.cat([tmp['preds'] for tmp in outputs])
            targets = torch.cat([tmp['target'] for tmp in outputs])
            cm = self.metrics.cm(preds, targets)
            plt.figure(figsize=(10, 7))
            fig_ = sns.heatmap(cm.cpu(), annot=True, fmt='.1f', cmap='coolwarm').get_figure()
            plt.close(fig_)
            self.logger.experiment.add_figure("cm/val/dataloader_idx_" +
                                              str(outputs[0]["dataloader_idx"]), fig_, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


my_model = MyModel(
    class_weights=class_weights,
    learning_rate=LEARNING_RATE,
    enable_mmd=ENABLE_MMD,
    alpha=ALPHA
)

from pytorch_lightning.callbacks import ModelSummary, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


logger = TensorBoardLogger(save_dir=".", log_graph=True)

callbacks = [
    ModelSummary(max_depth=2),
    EarlyStopping(monitor="total_loss/train", min_delta=0.00, patience=15, mode="min")
]

if torch.cuda.device_count() > 0:
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, accelerator="gpu", devices=1)
else:
    trainer = pl.Trainer(callbacks=callbacks, logger=logger)
trainer.fit(my_model, train_loader, val_dataloaders=[src_test_loader, trg_test_loader])

# %%
