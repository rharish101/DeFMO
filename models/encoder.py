import torch.nn as nn
import torchvision.models
from torch import Tensor

from utils import CkptModule


class EncoderCNN(CkptModule):
    def __init__(self) -> None:
        super().__init__()

        model = torchvision.models.resnet18(pretrained=True)
        layers = list(model.children())

        layer_0 = nn.Conv2d(
            6, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        layer_0.weight.data[:, :3, :, :] = nn.Parameter(layers[0].weight)
        layer_0.weight.data[:, 3:, :, :] = nn.Parameter(layers[0].weight)

        self.net = nn.Sequential(layer_0, *layers[1:3], *layers[4:8])

    def forward(self, inputs: Tensor) -> Tensor:
        return self.ckpt_run(self.net, 2, inputs)
