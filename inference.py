import os
import torch
import time
import torch.nn as nn
import datetime
import argparse
import warnings
import numpy as np
import torchvision.transforms as transforms

from autoencoder import AutoEncoder
from gan import Generator, Discriminator
from logger import create_logger
from omegaconf import OmegaConf
from itertools import chain
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets import PicmusTrainDataset, PicmusValDataset

from tqdm import tqdm


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


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device('cuda:{}'.format(args.gpu))


# Model: autoencoder, generator, discriminator
autoencoder = AutoEncoder()
generator = Generator(img_shape=(opt.channels, opt.img_size, opt.img_size))
discriminator = Discriminator()

sd = torch.load('logs/2023-03-15T21-33-05/checkpoints/autoencoder.pth')
autoencoder.load_state_dict(sd, strict=True)
sd = torch.load('logs/2023-03-15T21-33-05/checkpoints/generator.pth')
generator.load_state_dict(sd, strict=True)

# Set Device
autoencoder.to(device)
generator.to(device)


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


consume_time = 0
for batch in tqdm(dataloader):

    # Model inputs
    us_image = batch["us_image"].to(device)
    rf_data = batch["rf_data"].to(device)

    prev_time = time.time()
    encoder_rf,_ = autoencoder(rf_data)
    pred = generator(encoder_rf)
    consume_time += (time.time() - prev_time)

print(consume_time / len(dataloader) * 1000)
