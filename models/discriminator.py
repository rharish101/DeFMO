import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn.utils import spectral_norm
from torchvision.models import resnet18

from config import Config
from utils import CkptModule


def spectralize(module: nn.Module) -> nn.Module:
    if "weight" in module._parameters:
        return spectral_norm(module)
    for key, value in module._modules.items():
        module._modules[key] = spectralize(value)
    return module


class _FreezableModule(CkptModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trainable_params = [
            param for param in self.parameters() if param.requires_grad
        ]

    def freeze(self) -> None:
        self.eval()
        for param in self._trainable_params:
            param.requires_grad = False

    def unfreeze(self) -> None:
        self.train()
        for param in self._trainable_params:
            param.requires_grad = True


class Discriminator(_FreezableModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        model = resnet18(pretrained=True)
        layers = list(model.children())

        pretrained_weights = layers[0].weight
        layers[0] = nn.Conv2d(
            4,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        layers[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)

        combine = nn.Conv2d(512, 1, kernel_size=(1, 1))
        net = nn.Sequential(*layers[:-2], combine, layers[-2])
        self.net = spectralize(net)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with autocast(enabled=self.config.mixed_precision):
            noise = torch.randn_like(inputs) * 1e-3
            return self.ckpt_run(self.net, 2, inputs + noise).flatten()


class TemporalDiscriminator(_FreezableModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        model = resnet18(pretrained=True)
        layers = list(model.children())

        pretrained_weights = layers[0].weight
        layers[0] = nn.Conv2d(
            8,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        layers[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        layers[0].weight.data[:, 4:7, :, :] = nn.Parameter(pretrained_weights)

        classify = nn.Linear(512, 1)
        net = nn.Sequential(*layers[:-1], nn.Flatten(), classify)
        self.net = spectralize(net)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with autocast(enabled=self.config.mixed_precision):
            return self.ckpt_run(self.net, 2, inputs).flatten()
