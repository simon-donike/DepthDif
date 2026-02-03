# Imports
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import pytorch_lightning as pl

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
from skimage import io
import os

from models.difFF import *

# Load dataset


# Load MOdel

model=PixelDiffusionConditional(train_ds,
                                lr=1e-4,
                                batch_size=16)