import torch
import torch.nn as nn
import torchvision.models

from config import Config


class RenderingCNN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.net = self.get_model(config)
        self.rgba_operation = nn.Sigmoid()

    @staticmethod
    def get_model(config: Config) -> nn.Module:
        last_channels = 4
        if (
            config.use_selfsupervised_timeconsistency
            and config.timeconsistency_type == "oflow"
        ):
            last_channels += 2
        model = nn.Sequential(
            nn.Conv2d(
                513, 1024, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(
                1024,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(inplace=True),
            torchvision.models.resnet.Bottleneck(1024, 256),
            nn.PixelShuffle(2),
            torchvision.models.resnet.Bottleneck(256, 64),
            nn.PixelShuffle(2),
            torchvision.models.resnet.Bottleneck(64, 16),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                4, last_channels, kernel_size=3, stride=1, padding=1, bias=True
            ),
        )
        return model

    def forward(
        self, latent: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        renders_list = []
        shuffled_times_list = []

        for ki in range(times.shape[0]):
            shuffled_times_list.append(torch.randperm(times.shape[1]))
        shuffled_times = torch.stack(shuffled_times_list, 1).contiguous().T

        for ki in range(times.shape[1]):
            t_tensor = (
                times[range(times.shape[0]), shuffled_times[:, ki]]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, 1, latent.shape[2], latent.shape[3])
            )
            latenti = torch.cat((t_tensor, latent), 1)
            result = self.net(latenti)
            renders_list.append(result)

        renders = torch.stack(renders_list, 1).contiguous()
        renders[:, :, :4] = self.rgba_operation(renders[:, :, :4])
        for ki in range(times.shape[0]):
            renders[ki, shuffled_times[ki, :]] = renders[ki, :].clone()

        return renders
