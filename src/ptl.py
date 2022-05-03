# %%
%load_ext autoreload
%autoreload 2
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

INPUT_LENGTH = 128
BATCH_SZ = 256

# %%


def torch_prepare_dataloader(x_src, x_trg, y, batch_sz):
    x = torch.Tensor(x).unsqueeze(1)
    y = torch.Tensor(y).type(torch.LongTensor)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_sz, num_workers=8)


x_src, y_src = read_dataset(CWRUA, input_length=INPUT_LENGTH)
x_trg, y_trg = read_dataset(CWRUB, input_length=INPUT_LENGTH, cap_length=x_src.shape[0])

x_src = torch.Tensor(x_src).unsqueeze(1)
x_trg = torch.Tensor(x_trg).unsqueeze(1)
y_src = torch.Tensor(y_src).type(torch.LongTensor)
y_trg = torch.Tensor(y_trg).type(torch.LongTensor)
dataset = TensorDataset(x_src, x_trg, y_src)
train_loader = DataLoader(dataset, batch_size=BATCH_SZ, num_workers=8)
dataset = TensorDataset(x_trg, y_trg)
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


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

        )
        self.classifier = nn.Sequential(
            nn.Linear(480, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """in lightning, forward defines the prediction/inference actions"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # flatten all dimensions except batch
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x_src, x_trg, y_src = batch
        # Extract features
        x_src = self.encoder(x_src)
        x_trg = self.encoder(x_trg)
        # Classify
        x_src = x_src.view(x_src.size(0), -1)  # flatten all dimensions except batch
        y_hat = self.classifier(x_src)
        loss = F.cross_entropy(y_hat, y_src)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'preds': torch.argmax(y_hat, axis=1), 'target': y}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        cm = torchmetrics.ConfusionMatrix(num_classes=4, normalize="true")(preds, targets)
        # df_cm = pd.DataFrame(confusion_matrix.numpy(), index=range(4), columns=range(4))
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(cm.numpy(), annot=True, fmt='.1f', cmap='coolwarm').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("cm_val", fig_, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


autoencoder = LitAutoEncoder(learning_rate=1e-03)
# x, y = next(iter(train_loader))
# preds = autoencoder(x)
# preds.shape, y.shape

from pytorch_lightning.callbacks import ModelSummary, EarlyStopping


callbacks = [
    ModelSummary(max_depth=2),
    EarlyStopping(monitor="val_loss", min_delta=0.00, patience=15, mode="min")

]
trainer = pl.Trainer(callbacks=callbacks)
trainer.fit(autoencoder, train_loader, test_loader)

# %%
