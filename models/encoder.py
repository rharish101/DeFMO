from torch import Tensor
import torch.nn as nn
import torchvision.models

class EncoderCNN(nn.Module):
    def __init__(self) -> None:
        super(EncoderCNN, self).__init__()
        # self.net = self.get_v1()
        # self.net = self.get_v2()
        # self.net = self.get_v3()
        self.net = self.get_v4()

    def get_v1(self) -> nn.Module:
        model = torchvision.models.resnet50(pretrained=True)
        modelc = nn.Sequential(*list(model.children())[:-2])
        pretrained_weights = modelc[0].weight
        modelc[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        modelc[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        modelc[0].weight.data[:, 3:, :, :] = nn.Parameter(pretrained_weights)
        return nn.Sequential(modelc, nn.PixelShuffle(2))

    def get_v2(self) -> nn.Module:
        model = torchvision.models.resnet50(pretrained=True)
        modelc1 = nn.Sequential(*list(model.children())[:3])
        modelc2 = nn.Sequential(*list(model.children())[4:8])
        pretrained_weights = modelc1[0].weight
        modelc1[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        modelc1[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        modelc1[0].weight.data[:, 3:, :, :] = nn.Parameter(pretrained_weights)
        modelc = nn.Sequential(modelc1, modelc2)
        return modelc

    def get_v3(self) -> nn.Module:
        model = torchvision.models.resnet50(pretrained=True)
        modelc1 = nn.Sequential(*list(model.children())[:3])
        modelc2 = nn.Sequential(*list(model.children())[4:8])
        pretrained_weights = modelc1[0].weight
        modelc1[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        modelc1[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        modelc1[0].weight.data[:, 3:, :, :] = nn.Parameter(pretrained_weights)
        modelc3 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        modelc = nn.Sequential(modelc1, modelc2, modelc3)
        return modelc

    def get_v4(self) -> nn.Module:
        model = torchvision.models.resnet18(pretrained=True)
        layers = list(model.children())
        layer_0 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        layer_0.weight.data[:, :3, :, :] = nn.Parameter(layers[0].weight)
        layer_0.weight.data[:, 3:, :, :] = nn.Parameter(layers[0].weight)
        modelc = nn.Sequential(layer_0, *layers[1:3], *layers[4:8])
        return modelc

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)
