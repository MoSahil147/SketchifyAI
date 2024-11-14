# training/pix2pix_model.py
import torch
import torch.nn as nn

class Pix2PixGenerator(nn.Module):
    def __init__(self):
        super(Pix2PixGenerator, self).__init__()
        pass

    def forward(self, x):
        return x

class Pix2PixDiscriminator(nn.Module):
    def __init__(self):
        super(Pix2PixDiscriminator, self).__init__()
        pass

    def forward(self, x):
        return x