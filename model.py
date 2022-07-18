import torch
from torch import nn
import pytorch_lightning as pl
import seaborn as sns
import torchmetrics
import torch.nn.functional as F
import matplotlib

# Use headless backend for plotting
matplotlib.use("Agg")

# Choose which hparams to log in tensorboard
LOG_HPARAMS = ["learning_rate", "num_classes", "mmd_type", "alpha", "beta",
               "source", "target", "batch_size", "input_length", "test_size", "weight_decay", "reuse_target", "lr_patience", "lr_factor"]


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

class ResNextBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, in_sz):
        super().__init__()
        # if downsample:
        #     self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
        #         nn.BatchNorm2d(out_channels)
        #     )
        # else:
        self.shortcut = nn.Sequential()

        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=7, stride=1, padding=3, groups=in_channels)
        self.norm = nn.LayerNorm([in_sz])
        self.conv2 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, stride=1)

        # self.mainpass = nn.Sequential(
        #     nn.Conv1d(in_channels, in_channels, kernel_size=7, stride=1, padding=3, groups=in_channels),
        #     nn.LayerNorm([246]),
        #     nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1),
        #     nn.GELU(),
        #     nn.Conv1d(hidden_channels, out_channels, kernel_size=1, stride=1),
        # )

    def forward(self, input):
        shortcut = self.shortcut(input)
        x = self.norm(self.conv1(input))
        x =  nn.GELU()(self.conv2(x))
        x = self.conv3(x)
        return x + shortcut

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
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, stride=1),

            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, groups=32),
            # nn.LayerNorm([240]),
            # nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1, stride=1),
            # nn.GELU(),
            # nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, stride=1),
            ResNextBlock(in_channels=32, hidden_channels=32*4, out_channels=32, in_sz=246),
            nn.MaxPool1d(kernel_size=2),

            ResNextBlock(in_channels=32, hidden_channels=32*4, out_channels=32, in_sz=123),
            nn.MaxPool1d(kernel_size=2),

            ResNextBlock(in_channels=32, hidden_channels=32*4, out_channels=32, in_sz=61),
            nn.MaxPool1d(kernel_size=2),

            ResNextBlock(in_channels=32, hidden_channels=32*4, out_channels=32, in_sz=30),
            nn.MaxPool1d(kernel_size=2),

            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7),
            # # nn.BatchNorm1d(32),
            # nn.GELU(),
            # nn.MaxPool1d(kernel_size=2),

            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            # # nn.BatchNorm1d(32),
            # nn.GELU(),
            # nn.MaxPool1d(kernel_size=2),

            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            # # nn.BatchNorm1d(32),
            # nn.GELU(),
            # nn.MaxPool1d(kernel_size=2),
        )

        # if self.hparams.alpha > 0:
        self.mmd = MMD(kernel_type=self.hparams.mmd_type)
        if self.hparams.beta > 0:
            self.hloss = HLoss

        self.classifier = nn.Sequential(
            nn.Linear(480, 50),
            nn.GELU(),
            nn.Linear(50, 42),
            nn.GELU(),
            nn.Linear(42, self.hparams.num_classes),
        )
        # python trainer_main.py --accelerator gpu --gpus 1 --save_embeddings false --experiment_name baseline --lr_patience 30 --learning_rate 1e-3 --alpha 0 --beta 0
        self.softmax = nn.Softmax(dim=1)
        self.crossentropy_loss = nn.CrossEntropyLoss()
        # required to plot model computational graph on tensorboard
        self.example_input_array = torch.rand(self.hparams.batch_size, 1, self.hparams.input_length)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("MyModel")
        # HPARAMS
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--lr_factor", type=float, default=0.5)
        parser.add_argument("--lr_patience", type=int, default=10)
        parser.add_argument("--mmd_type", type=str, default="rbf")
        parser.add_argument("--alpha", type=float, default=1)
        parser.add_argument("--beta", type=float, default=0)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        # OTHER HPARAMS
        parser.add_argument("--num_classes", type=int, default=3)
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

        # if self.hparams.alpha > 0 or self.hparams.beta > 0:
        x_trg = self.encoder(x_trg)
        x_trg = x_trg.view(-1, x_trg.shape[1] * x_trg.shape[2])

        if self.hparams.alpha > 0:
            mmd_loss = self.mmd(x_src, x_trg)
        else:
            with torch.no_grad():
                mmd_loss = self.mmd(x_src, x_trg)
        self.log("mmd_loss/train", mmd_loss)
        # else:
        #     mmd_loss = 0.0

        if self.hparams.beta > 0:
            entropy_src = self.hloss(x_src)
            entropy_trg = self.hloss(x_trg)
            entropy_sum = entropy_src+entropy_trg
            self.log("entropy_src/train", entropy_src)
            self.log("entropy_trg/train", entropy_trg)
            self.log("entropy_sum/train", entropy_sum)
        else:
            entropy_sum = 0.0

        # Classify
        x_src = self.classifier(x_src)
        classif_loss = self.crossentropy_loss(x_src, y_src)
        total_loss = classif_loss + mmd_loss * self.hparams.alpha + entropy_sum * self.hparams.beta
        self.log("classificaiton_loss/train", classif_loss)
        self.log("total_loss/train", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y_true = batch
        x = self.encoder(x)
        embeddings = x.view(x.size(0), -1)
        x = self.classifier(embeddings)
        y_hat = self.softmax(x)  # onehot-encoded probabilities
        y_pred = torch.argmax(y_hat, axis=1)  # numeric preds
        # Compute and log metrics
        loss = self.crossentropy_loss(x, y_true)
        accu = self.metrics.accuracy(y_pred, y_true)
        self.log("accuracy/val", accu)
        self.log("classificaiton_loss/val", loss)

        # Log target accuracy as hyperparameter metric
        # this is very useful when combined with grid search
        # and tensorboard hparams tab
        if dataloader_idx == 1:
            self.log("hp_metric", accu, add_dataloader_idx=False)

        return {'loss': loss, 'y_pred': y_pred, 'y_true': y_true, 'embeddings': embeddings, 'dataloader_idx': dataloader_idx}

    def validation_epoch_end(self, dataloaders_outputs):
        dataloaders_embedd = []
        metadata = []

        for outputs in dataloaders_outputs:
            dataloader_name = "dataloader_idx_" + str(outputs[0]["dataloader_idx"])
            # Put togherer results from all batch steps in this dataloader
            y_true = torch.cat([tmp['y_true'] for tmp in outputs])
            y_pred = torch.cat([tmp['y_pred'] for tmp in outputs])
            embeddings = torch.cat([tmp['embeddings'] for tmp in outputs])

            # Confusion matrix
            cm = self.metrics.cm(y_pred, y_true)
            matplotlib.pyplot.figure(figsize=(18, 13))

            g = sns.heatmap(cm.cpu(), annot=True, fmt='.2f', cmap='coolwarm', annot_kws={'size': 25}, xticklabels=self.trainer.datamodule.class_lbls, yticklabels=self.trainer.datamodule.class_lbls)
            g.set_yticklabels(g.get_yticklabels(), rotation = 90, fontsize=25)
            g.set_xticklabels(g.get_xticklabels(), rotation = 0, fontsize=25)
            g.set_xlabel('Predicted class', fontsize=25, labelpad=35)
            g.set_ylabel('True class', fontsize=25, labelpad=35)
            g.collections[0].colorbar.ax.tick_params(labelsize=25)
            fig_ = g.get_figure()
            matplotlib.pyplot.close(fig_)
            self.logger.experiment.add_figure("cm/val/" + dataloader_name, fig_, self.current_epoch)

            # Pick a random subsample of embeddings for tensorboard plotting
            SUBSAMPLE = 64
            indices = torch.randperm(len(embeddings))[:SUBSAMPLE]
            dataloaders_embedd.append(embeddings[indices])
            metadata += [f'class{int(t)} set {dataloader_name}' for t in y_true[indices]]

        if self.save_embeddings:
            # Plotter
            self.logger.experiment.add_embedding(torch.cat(dataloaders_embedd), tag="embeddings", metadata=metadata)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
             verbose=True,
             patience=self.hparams.lr_patience,
             factor=self.hparams.lr_factor,)
        # reduce every epoch (default)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'total_loss/train'
        }
        return [optimizer], [scheduler]
