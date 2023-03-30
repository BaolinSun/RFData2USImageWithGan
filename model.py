import os
import torch
import torch.nn as nn

from torchvision import models
from collections import namedtuple

from utils import get_ckpt_path


class LPIPSWithDiscriminator(nn.Module):

    def __init__(self, disc_start, logvar_init=0.0, perceptual_weight=1.0, disc_conditional=False) -> None:
        super().__init__()

        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.disc_conditional = disc_conditional

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step,
                last_layer=None, cond=None, split='train', weights=None):
        rec_loss = torch.abs(inputs.contiguous() -
                             reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss

        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(
            weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_los = torch.sum(nll_loss) / nll_loss.shape[0]

        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                # assert not self.
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())


class DiagonalGaussianDistribution(object):

    def __init__(self, parameters, determinstic=False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 30.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.determinstic = determinstic
        if self.determinstic:
            self.var = self.std = torch.zeros_like(
                self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * \
            torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def mode(self):
        return self.mean

    def kl(self, other=None):
        if self.determinstic:
            return torch.Tensor([0.])

        if other is None:
            return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - self.logvar - 1.0, dim=[1, 2, 3])
        else:
            return 0.5 * torch.sum(torch.pow(self.mean - other.mean, 2)/other.var + self.var/other.var
                                   + other.logvar - self.logvar - 1.0, dim=[1, 2, 3])


class LPIPS(nn.Module):
    def __init__(self, use_dropout=True) -> None:
        super().__init__()

        self.scaling_layer = ScalingLayer
        self.chns = [64, 128, 256, 512, 512]    # vgg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)

        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name='vgg_lpips'):
        ckpt = get_ckpt_path(name, 'checkpoints/lpips')
        self.load_state_dict(torch.load(
            ckpt, map_location=torch.device('cpu')), strict=False)
        print('loaded pretrained LPIPS loss from {}'.format(ckpt))

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(
            input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.input(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(
                outs0[kk], normalize_tensor[outs1[kk]])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
               for kk in range(len(self.chns))]

        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]

        return val


class ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('shift', torch.Tensor(
            [-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor(
            [-.458, -.448, -.450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False) -> None:
        super().__init__()
        layers = [nn.Dropout(), ] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1,
                             padding=0, bias=False),]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        print(vgg_pretrained_features)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2,
                          h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


if __name__ == '__main__':
    net = vgg16(pretrained=False)
    # print(net)
