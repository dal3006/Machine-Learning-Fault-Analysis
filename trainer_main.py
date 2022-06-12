from argparse import ArgumentParser
from model import MyModel
from data import MyDataModule
from pytorch_lightning.callbacks import ModelSummary, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import pytorch_lightning as pl
import os

# PARSE ARGS
# ---

# Add args
parser = ArgumentParser()
parser = MyDataModule.add_argparse_args(parser)
parser = MyModel.add_argparse_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
logparser = parser.add_argument_group("MyLogger")
logparser.add_argument("--experiment_name", "-n", type=str, default="default")
parser.add_argument('--autorestore', default="true", type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
parser.add_argument('--restore', type=str)

# Read args
args = parser.parse_args()

# AUTORESTORE FOR GRID.AI SPOT INSTANCES
# ---
autorestore_file = None
if args.restore:
    autorestore_file = args.restore
elif args.autorestore and os.path.exists("./autorestore/last.ckpt"):
    print("autorestore param true and restore file found.")
    autorestore_file = "./autorestore/last.ckpt"


# LOAD DATA
# ---
data_module = MyDataModule.from_argparse_args(args)

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
    EarlyStopping(monitor="accuracy/val/dataloader_idx_1", min_delta=0.0, patience=60, mode="max"),
    # Save most recent weights on each epoch
    ModelCheckpoint(dirpath="./autorestore/", filename=".autorestore", save_last=True),
]

# START TRAINING
# ---

my_model = MyModel(**vars(args))
trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, max_time={"hours": 2})
trainer.fit(my_model, data_module, ckpt_path=autorestore_file)

# EVALUATE
# ---
best_accu = float(bestaccu_callback.best_model_score)
print(f'{best_accu=}')
