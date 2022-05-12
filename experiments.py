# %%
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import numpy as np

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

for i in range(10):
    lbls = np.random.choice(["a", "b", "c"], 64)
    writer.add_scalar("testtt", 42, i)
    writer.add_embedding(torch.rand(64, 128), tag="tag42", metadata=lbls, global_step=i)
writer.close()
