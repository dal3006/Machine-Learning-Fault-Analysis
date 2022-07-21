from argparse import ArgumentParser
from model import MyModel
from data import MyDataModule
from pytorch_lightning.callbacks import ModelSummary, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import os
import pandas as pd


# PARSE ARGS
# ---
# Add args
parser = ArgumentParser()
parser = MyDataModule.add_argparse_args(parser)
parser = MyModel.add_argparse_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
logparser = parser.add_argument_group("MyLogger")
logparser.add_argument("--experiment_name", "-n", type=str, default="default")
parser.add_argument('--resume_chkp_path', type=str, help="Path to a specific checkpoint to restore. Must exist.")
parser.add_argument('--resume_chkp_last', action='store_true', default=False, required=False, help="Restore last checkpoint if exists. Useful for resuming training on spot instances.")

# Read args
args = parser.parse_args()

# AUTORESTORE FOR GRID.AI SPOT INSTANCES
# ---
autorestore_file = None
if args.resume_chkp_path:
    print(f">>>>>>>>> Restoring from {args.resume_chkp_path} <<<<<<<<<")
    autorestore_file = args.resume_chkp_path
elif args.resume_chkp_last and os.path.exists("./autorestore/last.ckpt"):
    print(">>>>>>>>>> Restoring last checkpoint <<<<<<<<<<")
    autorestore_file = "./autorestore/last.ckpt"


# LOAD DATA
# ---
data_module = MyDataModule.from_argparse_args(args)


def train(args_dict):

    # CALLBACKS & LOGGERS
    # ---
    # Log to tensorboard
    logger = TensorBoardLogger(save_dir="./lightning_logs/", name=args.experiment_name, log_graph=True)
    # Save weights when model reaches best target accuracy
    bestaccu_callback = ModelCheckpoint(monitor="accuracy/val/dataloader_idx_1", mode="max", verbose=True)
    # Other callbacks
    callbacks = [
        bestaccu_callback,
        ModelSummary(max_depth=2),
        # Early stop if accuracy stops improving
        EarlyStopping(monitor="accuracy/val/dataloader_idx_1", min_delta=0.001, patience=20, mode="max"),
        # Save most recent weights on each epoch
        ModelCheckpoint(dirpath="./autorestore/", filename=".autorestore", save_last=True),
        LearningRateMonitor()
    ]

    # START TRAINING
    # ---
    my_model = MyModel(**args_dict)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, max_time={"hours": 2})
    trainer.fit(my_model, data_module, ckpt_path=autorestore_file)

    # EVALUATE
    # ---
    best_accu = float(bestaccu_callback.best_model_score)
    print(f'{best_accu=}')
    return best_accu


args_dict = vars(args)
train(args_dict)
