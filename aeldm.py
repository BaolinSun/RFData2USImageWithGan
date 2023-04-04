import os
import torch
import time
import torch.nn as nn
import datetime
import argparse
import warnings
import numpy as np
import torchvision.transforms as transforms

from autoencoder import RFAutoEncoder
from gan import Generator, Discriminator
from logger import create_logger
from omegaconf import OmegaConf
from itertools import chain
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets import PicmusTrainDataset, PicmusValDataset


# ================================Train config=================================

parser = argparse.ArgumentParser(description='us image generation parameters')
parser.add_argument(
    "-b",
    "--base",
    nargs="*",
    metavar="base_config.yaml",
    help="paths to base configs. Loaded from left-to-right. "
    "Parameters can be overwritten or added with command-line options of the form `--key value`.",
    default=list(),
)
parser.add_argument("-g", "--gpu", help="num of epochs", type=int, default=0)
args = parser.parse_args()
opt = OmegaConf.load(args.base[0])

now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
logdir = os.path.join('logs', now)
imagedir = os.path.join(logdir, 'images')
modeldir = os.path.join(logdir, 'checkpoints')
os.makedirs(logdir)
os.makedirs(imagedir)
os.makedirs(modeldir)
logger = create_logger(output_dir=logdir, name='')

# Tensor type
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device('cuda:{}'.format(args.gpu))


# ============================================================================

# configure dataloder
rf_transforms = [
    transforms.ToTensor(),
]
us_transforms = [
    transforms.ToTensor(),
    transforms.Resize((opt.img_size, opt.img_size)),
]

training_images_list_file = 'dataset/picmus/picmus_train.txt'
test_images_list_file = 'dataset/picmus/picmus_val.txt'
dataloader = DataLoader(
    PicmusTrainDataset(root='dataset/picmus', train_list_file=training_images_list_file, us_transforms=us_transforms, rf_transforms=rf_transforms),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    drop_last = True,
)

val_dataloader = DataLoader(
    PicmusValDataset(root='dataset/picmus', test_list_file=test_images_list_file, us_transforms=us_transforms, rf_transforms=rf_transforms),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=1,
)

# Model: autoencoder
autoencoder = RFAutoEncoder()

# Optimizers
optimizer_AE = torch.optim.Adam(autoencoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Loss functions
criterion_AE = torch.nn.MSELoss()

# Set Device
autoencoder.to(device)


for epoch in range(opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Model inputs
        us_image = batch["us_image"].to(device)
        rf_data = batch["rf_data"].to(device)

        batch_size = us_image.shape[0]

        encoder_rf, decoder_rf = autoencoder(rf_data)

