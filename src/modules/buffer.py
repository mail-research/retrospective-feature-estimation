from collections import deque
from typing import Any, Iterable, TypeVar

import attrs
import einops as ei
import numpy as np
import torch
import torch.utils.data as data_utils

T = TypeVar('T')


@attrs.define
class Buffer(data_utils.Dataset):
    max_size: int
    num_seen_samples: int = attrs.field(default=0, init=False)
    samples: deque[dict[str, Any]] = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.samples = deque(maxlen=self.max_size)

    @property
    def seen_idx(self):
        return self.num_seen_samples - 1

    @property
    def max_idx(self):
        return self.max_size - 1

    def __len__(self):
        return min(self.max_size, self.num_seen_samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
        return self.samples[idx]

    def is_full(self):
        return len(self.samples) == self.max_size

    def is_empty(self):
        return len(self.samples) == 0

    def empty(self):
        self.samples.clear()
        self.num_seen_samples = 0


def unbatch(**key_to_values: Iterable[T]) -> Iterable[dict[str, T]]:
    keys = key_to_values.keys()
    for batched_values in zip(*key_to_values.values()):
        sample = {key: value for key, value in zip(keys, batched_values)}
        yield sample


def process_sample(**key_to_value: T | Any) -> dict[str, T | Any]:
    return {
        key: ei.asnumpy(value) if isinstance(value, torch.Tensor) else value
        for key, value in key_to_value.items()
    }


def reservoir_modify(buffer: Buffer, seed: int = 0, **key_to_values: Iterable[T]):
    rng = np.random.default_rng(seed)
    for sample in unbatch(**key_to_values):
        if not buffer.is_full():
            buffer.samples.append(process_sample(**sample))
        else:
            idx = rng.integers(buffer.seen_idx + 1, endpoint=True)
            if idx < buffer.max_size:
                buffer.samples[idx] = process_sample(**sample)

        buffer.num_seen_samples += 1


def sample(buffer: Buffer, n_samples: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, len(buffer))
    idxs = rng.choice(len(buffer), size=n_samples, replace=False)
    return [buffer.samples[idx] for idx in idxs]
