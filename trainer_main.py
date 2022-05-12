from argparse import ArgumentParser
from model import MyModel
from data import MyDataModule
from pytorch_lightning.callbacks import ModelSummary, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import pytorch_lightning as pl
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

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
# args, unknown = parser.parse_known_args()

# LOAD DATA
data_module = MyDataModule.from_argparse_args(args)
data_module.prepare_data()
exit(0)


class CheckpointTracker(Callback):

    def __init__(self):
        self.collection = []

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self.collection.append(trainer.callback_metrics)


# PREPARE TRAINER
tracker = CheckpointTracker()
logger = TensorBoardLogger(save_dir=".", log_graph=True)
callbacks = [
    ModelSummary(max_depth=2),
    EarlyStopping(monitor="total_loss/train", min_delta=0.0, patience=15, mode="min"),
    ModelCheckpoint(monitor="accuracy/val/dataloader_idx_1", mode="max", verbose=True),
    tracker
]
trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)

# TRAIN
my_model = MyModel(class_weights=data_module.class_weights, **vars(args))
trainer.fit(my_model, data_module)

print("last_checkpoint_accu=" + str(float(tracker.collection[-1]['accuracy/val/dataloader_idx_1'])))
