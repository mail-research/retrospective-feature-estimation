from copy import deepcopy
from typing import Generic, TypeVar

import torch
import torch.nn as nn

from src.utils.misc import freeze

ModuleT = TypeVar('ModuleT', bound=nn.Module)


class Ema(nn.Module, Generic[ModuleT]):
    def __init__(self, module: ModuleT, alpha: float):
        super().__init__()
        self.module = deepcopy(module)
        self.alpha = alpha
        self.global_step = 0
        self.freeze()

    def freeze(self):
        freeze(self.module)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def step(self):
        self.global_step += 1

    def update(self, module: nn.Module):
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        for param, reference_param in zip(self.parameters(), module.parameters()):
            param.data.mul_(alpha).add_(reference_param.data, alpha=1 - alpha)
