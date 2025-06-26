import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, features=None):
        
        super().__init__()
        
        
        self.in_channeles = in_channels
        self.out_channels = out_channels
        self.features = features
        
        
        self.mlp = nn.Sequential(
                    nn.Linear(512*7*7, 512),
                    nn.LeakyReLU(True),
                    #nn.Dropout(p=0.3),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(True),
                    #nn.Dropout(p=0.3),
                    nn.Linear(256, 256),
                    nn.LeakyReLU(True),
                    #nn.Dropout(p=0.3),
                    nn.Linear(256, 128),
                    nn.LeakyReLU(True),
                    #nn.Dropout(p=0.3),
                    nn.Linear(128,self.out_channels) 
                )  

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512*7*7)
        output = self.mlp(x)
        
        return output
