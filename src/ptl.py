# %%
%load_ext autoreload
%autoreload 2
from gc import callbacks
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from eval import CWRUA, CWRUB, read_dataset
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torchmetrics

INPUT_LENGTH = 128
BATCH_SZ = 256

# %%


def torch_prepare_dataloader(x, y, batch_sz):
    x = torch.Tensor(x).unsqueeze(1)
    y = torch.Tensor(y).type(torch.LongTensor)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_sz, num_workers=8)


x, y = read_dataset(CWRUA, input_length=INPUT_LENGTH)
train_loader = torch_prepare_dataloader(x, y, batch_sz=BATCH_SZ)
x, y = read_dataset(CWRUB, input_length=INPUT_LENGTH)
test_loader = torch_prepare_dataloader(x, y, batch_sz=BATCH_SZ)

x, y = next(iter(train_loader))
print(x.shape)
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
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
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
