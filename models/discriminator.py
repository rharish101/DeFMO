import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchvision.models import resnet18


def spectralize(module: nn.Module) -> nn.Module:
    if 'weight' in module._parameters:
        return spectral_norm(module)
    for key, value in module._modules.items():
        module._modules[key] = spectralize(value)
    return module


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        model = resnet18(pretrained=True)
        layers = list(model.children())

        pretrained_weights = layers[0].weight
        layers[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        layers[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)

        combine = nn.Conv2d(512, 1, kernel_size=(1, 1))
        net = nn.Sequential(*layers[:-2], combine, layers[-2])
        self.net = spectralize(net)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(inputs) * 1e-3
        return self.net(inputs + noise).flatten()


class TemporalDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        model = resnet18(pretrained=True)
        layers = list(model.children())

        pretrained_weights = layers[0].weight
        layers[0] = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        layers[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        layers[0].weight.data[:, 4:7, :, :] = nn.Parameter(pretrained_weights)

        combine = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        net = nn.Sequential(*layers[:-2], combine, layers[-2])
        self.net = spectralize(net)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).flatten()
