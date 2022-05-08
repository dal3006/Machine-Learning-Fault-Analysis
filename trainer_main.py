from argparse import ArgumentParser
from model import MyModel
from data import MyDataModule
from pytorch_lightning.callbacks import ModelSummary, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl


# PARSE ARGS
parser = ArgumentParser()
# add PROGRAM level args
# parser.add_argument("--conda_env", type=str, default="some_name")
# parser.add_argument("--notification_email", type=str, default="will@email.com")
# add model specific args
parser = MyDataModule.add_argparse_args(parser)
parser = MyModel.add_argparse_args(parser)
# add all the available trainer options to argparse
# ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

# LOAD DATA
data_module = MyDataModule.from_argparse_args(args)
data_module.prepare_data()

# PREPARE TRAINER
logger = TensorBoardLogger(save_dir=".", log_graph=True)
callbacks = [
    ModelSummary(max_depth=2),
    EarlyStopping(monitor="total_loss/train", min_delta=0.01, patience=15, mode="min")
]
trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)

# TRAIN
my_model = MyModel(class_weights=data_module.class_weights, **vars(args))
trainer.fit(my_model, data_module)
