import torch
import torch.nn as nn
from torchvision.models import vgg19_bn
from collections import namedtuple
from utils import *


class percep_vgg19_bn(nn.Module):
    def __init__(self):
        super(percep_vgg19_bn, self).__init__()
        self.features = nn.Sequential(
            *list(vgg19_bn(pretrained=True).features.children())[:-1]
        )

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {5, 12, 25, 38, 51}:
                results.append(x)
        return results[0], results[1], results[2], results[3], results[4]

