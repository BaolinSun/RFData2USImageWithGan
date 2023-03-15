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

# Model: autoencoder, generator, discriminator
autoencoder = AutoEncoder()
generator = Generator(img_shape=(opt.channels, opt.img_size, opt.img_size))
discriminator = Discriminator()

# Optimizers
optimizer_G = torch.optim.Adam(params=chain(generator.parameters(),autoencoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
criterion_AE = torch.nn.MSELoss()

# Set Device
autoencoder.to(device)
generator.to(device)
discriminator.to(device)
criterion_GAN.to(device)
criterion_pixelwise.to(device)
criterion_AE.to(device)

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


def sampleImages(batches_done, train_real, train_fake):
    """Saves a generated sample from the validation set"""
    batch = next(iter(val_dataloader))

    us_image = batch["us_image"].to(device)
    rf_data = batch["rf_data"].to(device)
    
    encoder_rf,_ = autoencoder(rf_data)

    test_fake = generator(encoder_rf)

    img_sample_test = torch.cat((test_fake.data, us_image.data), -2)
    img_sample_train = torch.cat((train_fake.data,train_real.data), -2)

    save_image(img_sample_test, os.path.join(imagedir, f"/test_{batches_done}.png"), nrow=5, normalize=True)
    save_image(img_sample_train, os.path.join(imagedir, f"/train_{batches_done}.png"), nrow=5, normalize=True)

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_size // 2 ** 4, opt.img_size // 2 ** 4)

lambda_pixel = 100
ae_weight = 10

for epoch in range(opt.n_epochs):

    for i, batch in enumerate(dataloader):

        # Model inputs
        us_image = batch["us_image"].to(device)
        rf_data = batch["rf_data"].to(device)

        batch_size = us_image.shape[0]

        #Adversarial ground truths
        valid = torch.Tensor(np.ones((batch_size, *patch))).to(device)
        fake = torch.Tensor(np.zeros((batch_size, *patch))).to(device)

        # -----------------
        #  Train Generator And AutoEncoder
        # -----------------

        optimizer_G.zero_grad()

        # Gan loss
        encoder_rf,decoder_rf = autoencoder(rf_data)

        fake_imgs = generator(encoder_rf)
        pred_fake = discriminator(encoder_rf,fake_imgs)
        loss_GAN = criterion_GAN(pred_fake, valid)

        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_imgs, us_image)
        
        # Autoencoder loss
        loss_AE = criterion_AE(decoder_rf, rf_data)

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel + ae_weight * loss_AE

        loss_G.backward()
        optimizer_G.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(encoder_rf.detach(),us_image)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(encoder_rf.detach(),fake_imgs.detach())
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()


        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        if i % 10 == 0:

            train_info = "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, AE loss: %f, pixel: %f, adv: %f] ETA: %s" % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_AE.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            
            logger.info(train_info)

        if batches_done % opt.sample_interval == 0:
            sampleImages(batches_done, us_image, fake_imgs)





