import torch
import torch.nn as nn
from einops.layers.torch import Reduce


class FeatureExtractor(nn.Module):
    d_output: int


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.layers(x) + self.shortcut(x))


class Resnet18(FeatureExtractor):
    def __init__(self):
        super().__init__()

        def make_layer(in_planes: int, planes: int, blocks: int, stride: int):
            strides = [stride] + [1] * (blocks - 1)
            layers = []
            for stride in strides:
                layers.append(BasicBlock(in_planes, planes, stride))
                in_planes = planes
            return nn.Sequential(*layers)

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            make_layer(64, 64, blocks=2, stride=1),
            make_layer(64, 128, blocks=2, stride=2),
            make_layer(128, 256, blocks=2, stride=2),
            make_layer(256, 512, blocks=2, stride=2),
            Reduce('b c h w -> b c', 'mean'),
        )

    @property
    def d_output(self) -> int:  # type:ignore[override]
        return 512

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class ViTSmallP16(FeatureExtractor):
    def __init__(self):
        super().__init__()
        from timm import create_model

        self.model = create_model('vit_small_patch16_224', pretrained=True)
        self.model.reset_classifier(0)  # type: ignore[operator]
        self.linear = nn.Sequential(
            nn.Linear(self.model.embed_dim, 512),  # type: ignore[arg-type]
            nn.ReLU(),
        )

    @property
    def d_output(self) -> int:  # type:ignore[override]
        return 512

    def forward(self, x: torch.Tensor):
        return self.linear(self.model(x))
