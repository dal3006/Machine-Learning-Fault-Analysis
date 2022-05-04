# %%
# %load_ext autoreload
# %autoreload 2

from doctest import testfile
from gc import callbacks
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from eval import CWRUA, CWRUB, read_dataset
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torchmetrics

INPUT_LENGTH = 256
BATCH_SZ = 64
TRAIN_OVERLAP = 0.8
TEST_OVERLAP = 0.8
TEST_SIZE = 0.1
ENABLE_MMD = True
LEARNING_RATE = 1e-03

# %%
x_src, y_src, _, _ = read_dataset(CWRUA,
                                  test_size=0,
                                  input_length=INPUT_LENGTH,
                                  train_overlap=TRAIN_OVERLAP,
                                  test_overlap=TEST_OVERLAP)
x_trg_train, y_trg_train, x_trg_test, y_trg_test = read_dataset(CWRUB,
                                                                test_size=TEST_SIZE,
                                                                input_length=INPUT_LENGTH,
                                                                train_overlap=TRAIN_OVERLAP,
                                                                test_overlap=TEST_OVERLAP)
x_trg_train = x_trg_train[0:x_src.size(0)]

# y_src = y_src * 0  # TODO remove

dataset = TensorDataset(x_src, x_trg_train, y_src)
train_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SZ, num_workers=8)
dataset = TensorDataset(x_trg_test, y_trg_test)
test_loader = DataLoader(dataset, batch_size=BATCH_SZ, num_workers=8)

print("Train dataloader")
x1, x2, y = next(iter(train_loader))
print(x1.shape)
print(x2.shape)
print(y.shape)
print("Test dataloader")
x1, y = next(iter(test_loader))
print(x1.shape)
print(y.shape)

# %%

[x1, x2, y1] = next(iter(train_loader))
for x_s, x_t, y_s in zip(x1, x2, y1):
    if (y_s) == 3:
        plt.figure(figsize=(8, 1))
        plt.plot(x_s.squeeze(0))
        plt.show()
        # print(y_s)


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


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate, enable_mmd=True,
                 # Fake params, for hparams logging
                 input_length=INPUT_LENGTH,
                 batch_sz=BATCH_SZ,
                 train_overlap=TRAIN_OVERLAP,
                 test_overlap=TEST_OVERLAP,
                 test_size=TEST_SIZE,):
        super().__init__()
        self.save_hyperparameters()
        self.metrics = torch.nn.ModuleDict({'cm': torchmetrics.ConfusionMatrix(num_classes=4, normalize="true")})
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),
            # nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3),
            nn.ReLU(),
            # nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),

            # nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),

            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),

            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),

        )

        if self.hparams.enable_mmd:
            self.mmd = MMD_loss()

        self.classifier = nn.Sequential(
            nn.Linear(992, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        self.softmax = nn.Softmax(dim=1)
        self.crossentropy_loss = nn.CrossEntropyLoss()

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
            mmd_loss = self.mmd(x_src, x_trg)
        else:
            mmd_loss = 0.0
        # Classify
        x_src = self.classifier(x_src)
        classif_loss = self.crossentropy_loss(x_src, y_src)
        # classif_loss = F.cross_entropy(y_hat, y_src)

        total_loss = classif_loss + mmd_loss
        self.log("train_loss_class", classif_loss)
        self.log("train_loss_mmd", mmd_loss)
        self.log("train_loss_total", total_loss)
        self.log("hp_metric", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        loss = self.crossentropy_loss(x, y)
        y_hat = self.softmax(x)
        self.log("val_loss_class", loss, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'preds': torch.argmax(y_hat, axis=1), 'target': y}

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     loss = F.cross_entropy(y_hat, y)
    #     self.log("val_loss_class", loss, on_epoch=True, prog_bar=True)
    #     return {'loss': loss, 'preds': torch.argmax(y_hat, axis=1), 'target': y}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        cm = self.metrics.cm(preds, targets)
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(cm.cpu(), annot=True, fmt='.1f', cmap='coolwarm').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("cm_val", fig_, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


autoencoder = LitAutoEncoder(learning_rate=LEARNING_RATE, enable_mmd=ENABLE_MMD)

from pytorch_lightning.callbacks import ModelSummary, EarlyStopping


callbacks = [
    ModelSummary(max_depth=2),
    EarlyStopping(monitor="val_loss_class", min_delta=0.00, patience=15, mode="min")

]

if torch.cuda.device_count() > 0:
    trainer = pl.Trainer(callbacks=callbacks, accelerator="gpu", devices=1)
else:
    trainer = pl.Trainer(callbacks=callbacks)
trainer.fit(autoencoder, train_loader, test_loader)

# %%