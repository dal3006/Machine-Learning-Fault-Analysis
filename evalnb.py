# %%
from argparse import ArgumentParser
from model import MyModel
from data import MyDataModule
from pytorch_lightning.callbacks import ModelSummary, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# %%
model = MyModel.load_from_checkpoint("lightning_logs/version_24/checkpoints/epoch=0-step=109.ckpt")


# %%
data_module = MyDataModule.from_argparse_args({})
