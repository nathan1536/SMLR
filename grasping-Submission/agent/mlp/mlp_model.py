
from torch import nn
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import os

# class MLP(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#
#         """
#         :param in_channels: number of input channels to the first linear layer
#         :param out_channels: number of output channels for linear layers
#         """
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(in_features=in_channels, out_features=512),
#             # nn.ReLU(),
#             nn.LeakyReLU(),
#             # nn.Dropout(p=0.3),
#             nn.Linear(in_features=512, out_features=512),
#             # nn.ReLU(),
#             nn.LeakyReLU(),
#             nn.Linear(in_features=512, out_features=512),
#             # nn.ReLU(),
#             nn.LeakyReLU(),
#             # nn.Dropout(p=0.3),
#             nn.Linear(in_features=512, out_features=out_channels)
#         )
#
#     def forward(self, x):
#         return self.model(x)


class MLP_1(nn.Module):
    def __init__(self, in_channels, out_channels, features=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.LeakyReLU(True),
            # nn.Dropout(p=0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(True),
            # nn.Dropout(p=0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(True),
            # nn.Dropout(p=0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(True),
            # nn.Dropout(p=0.3),
            nn.Linear(512, self.out_channels)
        )

    def forward(self, x):
        # print(x)
        x = self.features(x * 255.0)
        output = self.mlp(x)

        return output

class MLP_2(nn.Module):
    def __init__(self, in_channels, out_channels, features=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.mlp = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.LeakyReLU(True),
            # nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(True),
            # nn.Dropout(p=0.3),
            nn.Linear(256, 256),
            nn.LeakyReLU(True),
            # nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(True),
            # nn.Dropout(p=0.3),
            nn.Linear(128, self.out_channels)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 7 * 7)
        output = self.mlp(x)

        return output

class R3M_FT(nn.Module):
    def __init__(self, r3m):
        """:param r3m: the original r3m model(called with load_r3m())"""
        super().__init__()
        self.model = r3m
        for name, param in self.model.named_parameters():
            if 'layer4' not in name:  # Freeze if not part of the selected layer
                param.requires_grad = False

    def forward(self, x):

        return self.model(x)
