from torch import Tensor
import torch.nn as nn
import torchvision.models

class EncoderCNN(nn.Module):
    def __init__(self) -> None:
        super(EncoderCNN, self).__init__()

        model = torchvision.models.resnet18(pretrained=True)
        layers = list(model.children())
        layer_0 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        layer_0.weight.data[:, :3, :, :] = nn.Parameter(layers[0].weight)
        layer_0.weight.data[:, 3:, :, :] = nn.Parameter(layers[0].weight)

        self.net = nn.Sequential(layer_0, *layers[1:3], *layers[4:8])

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)
