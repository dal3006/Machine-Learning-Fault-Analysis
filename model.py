from importlib.metadata import metadata
from typing import List
import torch
from torch import nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import torchmetrics
import torch.nn.functional as F

# Choose which hparams to log in tensorboard
LOG_HPARAMS = ["learning_rate", "num_classes", "mmd_type", "alpha", "beta",
               "source", "target", "batch_size", "input_length", "test_size", "weight_decay", "reuse_target"]


class MMD(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]
            loss = torch.mean(XX + YY - XY - YX)
            return loss

def HLoss(x):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = -1.0 * b.sum()
    return b

class MyModel(pl.LightningModule):
    def __init__(self, save_embeddings, **kwargs):
        super().__init__()
        self.save_embeddings = save_embeddings
        self.save_hyperparameters(*LOG_HPARAMS)
        self.metrics = torch.nn.ModuleDict({
            'cm': torchmetrics.ConfusionMatrix(num_classes=self.hparams.num_classes, normalize="true"),
            'accuracy': torchmetrics.Accuracy()
        })
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            # nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            # nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            # nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            # nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
        )

        if self.hparams.alpha > 0:
            self.mmd = MMD(kernel_type=self.hparams.mmd_type)
        if self.hparams.beta > 0:
            self.hloss = HLoss

        self.classifier = nn.Sequential(
            nn.Linear(448, 32),
            nn.GELU(),
            nn.Linear(32, self.hparams.num_classes),
        )
        self.softmax = nn.Softmax(dim=1)
        self.crossentropy_loss = nn.CrossEntropyLoss()
        # required to plot model computational graph on tensorboard
        self.example_input_array = torch.rand(self.hparams.batch_size, 1, self.hparams.input_length)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("MyModel")
        # HPARAMS
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--mmd_type", type=str, default="rbf")
        parser.add_argument("--alpha", type=float, default=0.01)
        parser.add_argument("--beta", type=float, default=0.0001)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        # OTHER HPARAMS
        parser.add_argument("--num_classes", type=int, default=4)
        parser.add_argument("--save_embeddings", default="false",
                            type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
        return parent_parser

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
        x_src = x_src.view(-1, x_src.shape[1] * x_src.shape[2])  # flatten all dimensions except batch

        if self.hparams.alpha > 0 or self.hparams.beta > 0:
            x_trg = self.encoder(x_trg)
            x_trg = x_trg.view(-1, x_trg.shape[1] * x_trg.shape[2])

        if self.hparams.alpha > 0:
            mmd_loss = self.mmd(x_src, x_trg) * self.hparams.alpha
            self.log("mmd_loss/train", mmd_loss)
        else:
            mmd_loss = 0.0

        if self.hparams.beta > 0:
            entropy_src = self.hloss(x_src)
            entropy_trg = self.hloss(x_trg)
            entropy_sum = (entropy_src+entropy_trg) * self.hparams.beta
            self.log("entropy_src/train", entropy_src)
            self.log("entropy_trg/train", entropy_trg)
            self.log("entropy_sum/train", entropy_sum)
        else:
            entropy_sum = 0.0

        # Classify
        x_src = self.classifier(x_src)
        classif_loss = self.crossentropy_loss(x_src, y_src)
        total_loss = classif_loss + mmd_loss + entropy_sum
        self.log("classificaiton_loss/train", classif_loss)
        self.log("total_loss/train", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        x = self.encoder(x)
        embeddings = x.view(x.size(0), -1)

        x = self.classifier(embeddings)
        loss = self.crossentropy_loss(x, y)
        y_hat = self.softmax(x)  # onehot-encoded probabilities
        preds = torch.argmax(y_hat, axis=1)  # numeric preds
        # Compute and log metrics
        accu = self.metrics.accuracy(preds, y)
        self.log("classificaiton_loss/val", loss, on_epoch=True)
        self.log("accuracy/val", accu, on_epoch=True)
        return {'loss': loss, 'preds': preds, 'target': y, 'embeddings': embeddings, 'dataloader_idx': dataloader_idx}

    def validation_epoch_end(self, dataloaders_outputs):
        dataloaders_embedd = []
        metadata = []

        for outputs in dataloaders_outputs:
            dataloader_name = "dataloader_idx_" + str(outputs[0]["dataloader_idx"])
            preds = torch.cat([tmp['preds'] for tmp in outputs])
            targets = torch.cat([tmp['target'] for tmp in outputs])
            embeddings = torch.cat([tmp['embeddings'] for tmp in outputs])

            # Confusion matrix
            cm = self.metrics.cm(preds, targets)
            plt.figure(figsize=(10, 7))
            fig_ = sns.heatmap(cm.cpu(), annot=True, fmt='.2f', cmap='coolwarm').get_figure()
            plt.close(fig_)
            self.logger.experiment.add_figure("cm/val/" + dataloader_name, fig_, self.current_epoch)

            # Pick a random subsample
            SUBSAMPLE = 64
            indices = torch.randperm(len(embeddings))[:SUBSAMPLE]
            dataloaders_embedd.append(embeddings[indices])
            metadata += [f'class{int(t)} set {dataloader_name}' for t in targets[indices]]

        if self.save_embeddings:
            # Plotter
            self.logger.experiment.add_embedding(torch.cat(dataloaders_embedd), tag="embeddings", metadata=metadata)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer
