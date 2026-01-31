from functools import partial

import numpy as np
import torch


def new_seed(seed_or_rng: int | np.random.Generator = 0):
    return int(np.random.default_rng(seed_or_rng).integers(np.iinfo(np.int64).max))


def set_grad_mode(module: torch.nn.Module, requires_grad: bool = True):
    for param in module.parameters():
        param.requires_grad = requires_grad
    return module


freeze = partial(set_grad_mode, requires_grad=False)
unfreeze = partial(set_grad_mode, requires_grad=True)


def get_device(module_or_tensor: torch.nn.Module | torch.Tensor | None = None) -> torch.device:
    match module_or_tensor:
        case torch.nn.Module():
            return next(module_or_tensor.parameters()).device
        case torch.Tensor():
            return module_or_tensor.device
        case None:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        case _:
            raise TypeError
