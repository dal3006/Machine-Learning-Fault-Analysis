# %%
%load_ext autoreload
%autoreload 2
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl


# %%
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(1, 32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softmax(),
        )

    def forward(self, x):
        """in lightning, forward defines the prediction/inference actions"""
        x = self.encoder(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        y_hat = self.classifier(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# %%
from eval import CWRUA, CWRUB, read_dataset
from model import INPUT_LENGTH
from torch.utils.data import TensorDataset, DataLoader


def torch_prepare_dataloader(x, y):
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    x = torch.rand((512, 1, 128))
    y = torch.rand((512, 1, 4))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32)


x, y = read_dataset(CWRUA, input_length=INPUT_LENGTH)
train_loader = torch_prepare_dataloader(x, y)
x, y = read_dataset(CWRUB, input_length=INPUT_LENGTH)
test_loader = torch_prepare_dataloader(x, y)

x, y = next(iter(train_loader))
print(x.shape)
print(y.shape)

# %%

autoencoder = LitAutoEncoder()
trainer = pl.Trainer()
trainer.fit(autoencoder, train_loader, test_loader)

# %%

# %%
