from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Protocol,
    TypeVar,
    dataclass_transform,
    runtime_checkable,
)

import attrs
import einops as ei
import jaxtyping as jty
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torch.utils.data as data_utils

from src.modules.buffer import Buffer, reservoir_modify, sample
from src.modules.ema import Ema
from src.modules.net import FeatureExtractor, Resnet18
from src.utils.misc import get_device, new_seed

ImageT = jty.Float[torch.Tensor, 'b c h w']
TargetT = jty.Int[torch.Tensor, 'b']
VectorT = jty.Float[torch.Tensor, 'b d']

DatasetT = data_utils.Dataset[tuple[ImageT, TargetT]]


@dataclass_transform(field_specifiers=(attrs.field,))
def define_module(cls: type[nn.Module]):
    if not hasattr(cls, '__attrs_pre_init__'):
        cls.__attrs_pre_init__ = lambda self: nn.Module.__init__(self)  # type: ignore

    return attrs.define(cls, slots=False, repr=False, eq=False)


class TaskStartMixin(ABC):
    @abstractmethod
    def start_task(self, tkey: str, training_set: DatasetT, validation_set: DatasetT): ...


class TaskEndMixin(ABC):
    @abstractmethod
    def end_task(self, tkey: str, training_set: DatasetT, validation_set: DatasetT): ...


@attrs.frozen
class Output:
    logits: VectorT
    targets: TargetT | None = None
    loss: float | None = None


class ClassifierModuleDict(nn.ModuleDict):
    def forward(self, features: VectorT):
        return ei.pack([classifier(features) for classifier in self.values()], 'b *')[0]


ArgsT = TypeVar('ArgsT')


@define_module
class Method(nn.Module, ABC, Generic[ArgsT]):
    args: ArgsT

    extractor: FeatureExtractor = attrs.field(factory=partial(Resnet18, d_output=512))  # type: ignore[call-arg]
    scale_fn: nn.Module = attrs.field(factory=nn.Identity)
    augment_fn: nn.Module = attrs.field(factory=nn.Identity)
    head_constructor: Callable[[int, int], nn.Module] = attrs.field(default=nn.Linear)

    tkey_to_classifier: ClassifierModuleDict = attrs.field(init=False, factory=ClassifierModuleDict)
    tkey_to_d_output: dict[str, int] = attrs.field(init=False, factory=dict)

    fixed: bool = attrs.field(init=False, default=False)
    seen_tkeys: list[str] = attrs.field(init=False, factory=list)

    def add_task(self, tkey: str, d_output: int):
        if self.fixed:
            raise RuntimeError

        classifier = self.sync_device(self.head_constructor(self.extractor.d_output, d_output))
        self.tkey_to_classifier[tkey] = classifier
        self.tkey_to_d_output[tkey] = d_output

    def see_task(self, tkey: str):
        self.seen_tkeys.append(tkey)

    def make_optimizer_with_scheduling(
        self,
        lr: float,
        parameters: Iterable[torch.nn.Parameter] | None = None,
        optim_kwargs: dict[str, Any] | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
    ) -> tuple[optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau]:
        parameters = self.parameters() if parameters is None else parameters
        optim_kwargs = optim_kwargs or {}
        scheduler_kwargs = scheduler_kwargs or {}

        optimizer = optim.Adam(parameters, lr, **optim_kwargs)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
        return optimizer, scheduler

    def sync_device(self, *args: torch.Tensor | nn.Module):
        device = get_device(self.extractor)
        args = tuple(arg.to(device) for arg in args)
        return args[0] if len(args) == 1 else args

    def retarget(self, targets: TargetT, tkey: str):
        unused_units = 0
        for key, d_output in self.tkey_to_d_output.items():
            if key == tkey:
                break
            unused_units += d_output
        return targets - unused_units

    @abstractmethod
    def step(
        self,
        images: ImageT,
        targets: TargetT,
        tkey: str,
        optimizer: optim.Optimizer,
    ) -> float: ...

    @torch.inference_mode()
    def infer_logits(self, images: ImageT, tkey: str | None = None) -> Output:
        images = self.sync_device(images)
        processed_images = self.scale_fn(images)
        if tkey is not None:
            logits = self.tkey_to_classifier[tkey](self.extractor(processed_images))
        else:
            logits = self.tkey_to_classifier(self.extractor(processed_images))
        return Output(logits=logits)

    @torch.inference_mode()
    def infer(
        self,
        images: ImageT,
        targets: TargetT | None = None,
        tkey: str | None = None,
    ) -> Output:
        outputs = self.infer_logits(images, tkey=tkey)
        if targets is None:
            return outputs

        targets = self.sync_device(targets)
        if tkey is not None:
            targets = self.retarget(targets, tkey=tkey)
        loss = fn.cross_entropy(outputs.logits, targets).item()
        return attrs.evolve(outputs, targets=targets, loss=loss)


@attrs.frozen
class BufferArgs:
    buffer_size: int
    batch_size: int
    seed: int


BufferArgsT = TypeVar('BufferArgsT', bound=BufferArgs)


@runtime_checkable
class ModifyBufferFnT(Protocol):
    def __call__(self, buffer: Buffer, seed: int, **key_to_values: Iterable): ...


@define_module
class BufferMethod(Method[BufferArgsT]):
    buffer: Buffer = attrs.field(init=False)
    rng: np.random.Generator = attrs.field(init=False)
    modify_fn: ModifyBufferFnT = reservoir_modify

    @buffer.default  # type: ignore[attr-defined]
    def _buffer_default(self):
        return Buffer(self.args.buffer_size)

    @rng.default  # type: ignore[attr-defined]
    def _rng_default(self):
        return np.random.default_rng(self.args.seed)

    def sample_buffer(self, buffer: Buffer | None = None, n_samples: int | None = None):
        if buffer is None:
            buffer = self.buffer
        if n_samples is None:
            n_samples = self.args.batch_size
        return sample(buffer, n_samples=n_samples, seed=new_seed(self.rng))

    def sample_collate_buffer(self, buffer: Buffer | None = None, n_samples: int | None = None):
        return data_utils.default_collate(self.sample_buffer(buffer=buffer, n_samples=n_samples))

    def modify_buffer(self, buffer: Buffer | None = None, **key_to_values):
        if buffer is None:
            buffer = self.buffer
        self.modify_fn(buffer, seed=new_seed(self.rng), **key_to_values)


@define_module
class ReservoirBufferMethod(BufferMethod[BufferArgsT]):
    modify_fn: ModifyBufferFnT = attrs.field(init=False, default=reservoir_modify)


class TrackerMixin:  # separate from Method for simplicity
    seen_tkeys: list[str]
    tkey_to_d_output: dict[str, int]

    @property
    def n_seen_tasks(self) -> int:
        return len(self.seen_tkeys)

    @property
    def learned_tkeys(self):
        return self.seen_tkeys[:-1]

    @property
    def n_learned_tasks(self) -> int:
        return len(self.learned_tkeys)

    @property
    def current_tkey(self):
        if len(self.seen_tkeys) == 0:
            return None
        return self.seen_tkeys[-1]

    @property
    def last_tkey(self):
        if len(self.learned_tkeys) == 0:
            return None
        return self.learned_tkeys[-1]

    @property
    def n_seen_target_classes(self) -> int:
        return sum(self.tkey_to_d_output[tkey] for tkey in self.seen_tkeys)

    @property
    def n_learned_target_classes(self) -> int:
        return sum(self.tkey_to_d_output[tkey] for tkey in self.learned_tkeys)


@define_module
class EmaCopy(nn.Module):
    extractor: Ema[FeatureExtractor]
    tkey_to_classifier: Ema[ClassifierModuleDict]

    frequency: float
    seed: int = 0

    rng: np.random.Generator = attrs.field(init=False)

    @rng.default  # type: ignore[attr-defined]
    def _rng_default(self):
        return np.random.default_rng(self.seed)

    @classmethod
    def from_method(cls, method: Method, alpha: float, frequency: float, seed: int = 0):
        return cls(
            extractor=Ema(method.extractor, alpha=alpha),
            tkey_to_classifier=Ema(method.tkey_to_classifier, alpha=alpha),
            frequency=frequency,
            seed=seed,
        )

    def sync_method(self, method: Method):
        for tkey in method.tkey_to_classifier:
            if tkey in self.tkey_to_classifier.module:
                continue
            classifier = method.tkey_to_classifier[tkey]
            self.tkey_to_classifier.module[tkey] = deepcopy(classifier)
        self.tkey_to_classifier.freeze()

    def step(self, method: Method):
        self.extractor.step()
        self.tkey_to_classifier.step()
        if self.rng.random() <= self.frequency:
            self.extractor.update(method.extractor)
            self.tkey_to_classifier.update(method.tkey_to_classifier)
