from argparse import ArgumentParser
from model import MyModel
from data import MyDataModule
from pytorch_lightning.callbacks import ModelSummary, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import pytorch_lightning as pl
import os

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
logparser = parser.add_argument_group("MyLogger")
logparser.add_argument("--experiment_name", "-n", type=str, default="default")
parser.add_argument('--autorestore', default="true", type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
parser.add_argument('--restore', type=str)
args = parser.parse_args()

# restore checkpoint

# LOAD DATA
data_module = MyDataModule.from_argparse_args(args)


# class CheckpointTracker(Callback):

#     def __init__(self):
#         self.collection = []

#     def on_save_checkpoint(self, trainer, pl_module, checkpoint):
#         self.collection.append(trainer.callback_metrics)


# PREPARE TRAINER
# tracker = CheckpointTracker()
logger = TensorBoardLogger(save_dir="./lightning_logs/", name=args.experiment_name, log_graph=True)
callbacks = [
    ModelSummary(max_depth=2),
    EarlyStopping(monitor="accuracy/val/dataloader_idx_1", min_delta=0.0, patience=60, mode="max"),
    ModelCheckpoint(monitor="accuracy/val/dataloader_idx_1", mode="max",
                    verbose=True),  # save best target accuracy checkpoint
    ModelCheckpoint(dirpath="./autorestore/", filename=".autorestore", save_last=True),
]

autorestore_file = None
if args.restore:
    autorestore_file = args.restore
elif args.autorestore and os.path.exists("./autorestore/last.ckpt"):
    print("autorestore param true and restore file found.")
    autorestore_file = "./autorestore/last.ckpt"

trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, max_epochs=300, max_time={"hours": 2})

# TRAIN
my_model = MyModel(**vars(args))
trainer.fit(my_model, data_module, ckpt_path=autorestore_file)


# print("last_checkpoint_accu=" + str(float(tracker.collection[-1]['accuracy/val/dataloader_idx_1'])))
