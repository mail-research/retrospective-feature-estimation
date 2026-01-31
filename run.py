import os
from contextlib import suppress
from enum import Enum
from typing import Annotated, Any

import attrs
import cyclopts as cli
import einops as ei
import jaxtyping as jty
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.transforms.v2 as tf
from beartype import beartype
from loguru import logger
from tqdm import tqdm

from src.datasets import MetaDataset, make_task
from src.methods import Method
from src.modules.net import FeatureExtractor
from src.utils.logging import Logger
from src.utils.training import Setting, TrainingArgs

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True)

torch.set_num_threads(16)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.use_deterministic_algorithms(True, warn_only=False)

DEFAULT_N_EPOCHS = 40
DEFAULT_LR = 5e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 0
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DatasetOption(str, Enum):
    CIFAR10 = 'cifar10'
    CIFAR100_5S = 'cifar100-5s'
    CIFAR100_10S = 'cifar100-10s'
    TINY_IMAGENET_10S = 'tiny_imagenet-10s'
    TINY_IMAGENET_20S = 'tiny_imagenet-20s'

    def build(self, root: str) -> tuple[MetaDataset, dict[str, list[int]]]:
        def split(n_tasks: int, n_classes: int) -> dict[str, list[int]]:
            split_matrix = ei.rearrange(np.arange(n_classes), '(t c) -> t c', t=n_tasks)
            return {str(idx): arr.tolist() for idx, arr in enumerate(split_matrix, start=1)}  # type: ignore[misc]

        match self:
            case self.CIFAR10:
                from src.datasets.cifar10 import init

                tkey_to_target_classes = split(5, 10)

            case self.CIFAR100_5S:
                from src.datasets.cifar100 import init

                tkey_to_target_classes = split(5, 100)

            case self.CIFAR100_10S:
                from src.datasets.cifar100 import init

                tkey_to_target_classes = split(10, 100)

            case self.TINY_IMAGENET_10S:
                from src.datasets.tiny_imagenet import init

                tkey_to_target_classes = split(10, 200)

            case self.TINY_IMAGENET_20S:
                from src.datasets.tiny_imagenet import init

                tkey_to_target_classes = split(20, 200)

            case _:
                raise ValueError

        return init(root=root), tkey_to_target_classes


class BackboneOption(str, Enum):
    RESNET18 = 'resnet18'
    VIT_SMALL_P16 = 'vit-small-p16'

    def build(self) -> FeatureExtractor:
        match self:
            case self.RESNET18:
                from src.modules.net import Resnet18

                return Resnet18()

            case self.VIT_SMALL_P16:
                from src.modules.net import ViTSmallP16

                return ViTSmallP16()

            case _:
                raise ValueError


class LearningOption(str, Enum):
    FIXED = 'fixed'
    EXPANDING = 'expanding'
    JOINT = 'joint'

    def build(self):
        match self:
            case self.FIXED:
                from src.utils.training import learn_fixed

                return learn_fixed

            case self.EXPANDING:
                from src.utils.training import learn_expanding

                return learn_expanding

            case self.JOINT:
                from src.utils.training import learn_joint

                return learn_joint

            case _:
                raise ValueError


class LogOption(str, Enum):
    NONE = 'none'
    WANDB = 'wandb'
    CLEARML = 'clearml'

    def build(
        self,
        run_name: str,
        method_name: str,
        method_args: dict[str, Any],
        dataset_name: str,
        dataset_args: dict[str, Any],
    ) -> Logger:
        match self:
            case self.NONE:
                from src.utils.logging import Logger

                return Logger(
                    run_name=run_name,
                    method_name=method_name,
                    method_args=method_args,
                    dataset_name=dataset_name,
                    dataset_args=dataset_args,
                )

            case self.WANDB:
                raise NotImplementedError

            case self.CLEARML:
                from src.utils.logging import ClearMLLogger

                return ClearMLLogger(
                    run_name=run_name,
                    method_name=method_name,
                    method_args=method_args,
                    dataset_name=dataset_name,
                    dataset_args=dataset_args,
                )

            case _:
                raise ValueError


app = cli.App()


@beartype
def accuracy(logits: jty.Float[np.ndarray, 'b c'], targets: jty.Int[np.ndarray, 'b']) -> float:
    return (logits.argmax(axis=-1) == targets).mean().item()


@app.meta.default
@logger.catch
def run(
    *tokens: Annotated[str, cli.Parameter(show=False, allow_leading_hyphen=True)],
    dataset: DatasetOption,
    backbone: BackboneOption = BackboneOption.RESNET18,
    learning: LearningOption = LearningOption.FIXED,
    setting: Setting = Setting.TASK_IL,
    log: LogOption = LogOption.NONE,
    seed: int = DEFAULT_SEED,
    n_epochs: int = DEFAULT_N_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    toy: bool = False,
):
    method_class: type[Method]
    method_class, method_args = app(tokens)

    with suppress(TypeError):  # if method requires random seed, change to current random seed
        method_args = attrs.evolve(method_args, seed=seed)
    with suppress(TypeError):  # if method requires batch size, change to current batch size
        method_args = attrs.evolve(method_args, batch_size=batch_size)

    meta_dataset, tkey_to_target_classes = dataset.build(root='data')
    if backbone == BackboneOption.VIT_SMALL_P16:
        meta_dataset = attrs.evolve(
            meta_dataset,
            scale_fn=tf.Compose([tf.Resize(224), meta_dataset.scale_fn]),
        )

    logger.info(f'Using {meta_dataset.name} with splits {tkey_to_target_classes}')

    run_args = [
        method_class.__name__.lower(),
        setting.value,
        dataset.value,
        backbone.value,
        learning.value,
        f'seed-{seed}',
    ]

    if toy:
        training_set, test_set, _ = data_utils.random_split(
            meta_dataset.training_set,
            lengths=[0.2, 0.1, 0.7],
            generator=torch.Generator().manual_seed(seed),
        )
        meta_dataset = attrs.evolve(meta_dataset, training_set=training_set, test_set=test_set)
        with suppress(TypeError, AttributeError):
            method_args = attrs.evolve(method_args, buffer_size=int(method_args.buffer_size * 0.2))

        run_args.append('toy')

    run_name = '_'.join(run_args)

    experiment_logger = log.build(
        run_name=run_name,
        method_name=method_class.__name__,
        method_args=attrs.asdict(method_args),
        dataset_name=dataset.value,
        dataset_args={'task_splits': tkey_to_target_classes},
    )

    logger.info(f'Using {method_class.__name__} with args: {attrs.asdict(method_args)}')

    torch.manual_seed(seed)
    learn_fn = learning.build()
    feature_extractor = backbone.build().to(device)
    logger.info(f'Params: {sum(p.numel() for p in feature_extractor.parameters())}')
    logger.info(f'Hidden dim: {feature_extractor.d_output}')

    tkey_to_task = make_task(
        training_set=meta_dataset.training_set,
        test_set=meta_dataset.test_set,
        tkey_to_target_classes=tkey_to_target_classes,
    )

    args = TrainingArgs(
        batch_size=batch_size,
        seed=seed,
        setting=setting,
        lr=lr,
        n_epochs=n_epochs,
        validation_ratio=0.1,
        scheduler_kwargs={'patience': 3, 'factor': 0.1},
    )
    method = method_class(
        args=method_args,
        extractor=feature_extractor,
        scale_fn=meta_dataset.scale_fn,
        augment_fn=meta_dataset.augment_fn,
    )

    for tkey, task_result in learn_fn(
        method=method,
        tkey_to_task=tkey_to_task,
        args=args,
        metric_name_to_fn={'accuracy': accuracy},
    ):
        experiment_logger.log(tkey, task_result)
    experiment_logger.finish()


@app.command
def basic(reset_on_new_task: bool = False):
    from src.methods.basic import BasicArgs, BasicMethod

    return BasicMethod, BasicArgs(reset_on_new_task=reset_on_new_task)


@app.command
def er(buffer_size: int):
    from src.methods.er import ErArgs, ErMethod

    return ErMethod, ErArgs(
        buffer_size=buffer_size,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=DEFAULT_SEED,
    )


@app.command
def derpp(buffer_size: int, distill_weight: float, replay_weight: float):
    from src.methods.derpp import DerppArgs, DerppMethod

    return DerppMethod, DerppArgs(
        buffer_size=buffer_size,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=DEFAULT_SEED,
        distill_weight=distill_weight,
        replay_weight=replay_weight,
    )


@app.command
def er_ace(buffer_size: int):
    from src.methods.er_ace import ErAceArgs, ErAceMethod

    return ErAceMethod, ErAceArgs(
        buffer_size=buffer_size,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=DEFAULT_SEED,
    )


@app.command
def agem(buffer_size: int):
    from src.methods.agem import AgemArgs, AgemMethod

    return AgemMethod, AgemArgs(
        buffer_size=buffer_size,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=DEFAULT_SEED,
    )


@app.command
def rfe(
    regularize_weight: float,
    d_project: int = 128,
    d_aux: int = 128,
    n_module_epochs: int = DEFAULT_N_EPOCHS,
    module_lr: float = 5e-3,
    e2e: bool = False,
):
    from src.methods.rfe import RfeArgs

    if not e2e:
        from src.methods.rfe import RfeMethod
    else:
        from src.methods.rfe import RfeE2EMethod as RfeMethod  # type: ignore[assignment]

    return RfeMethod, RfeArgs(
        batch_size=DEFAULT_BATCH_SIZE,
        seed=DEFAULT_SEED,
        regularize_weight=regularize_weight,
        d_project=d_project,
        d_aux=d_aux,
        n_module_epochs=n_module_epochs,
        module_lr=module_lr,
    )


@app.command
def rfe_data(
    buffer_size: int,
    regularize_weight: float,
    only_last_task: bool,
    d_project: int = 128,
    d_aux: int = 128,
    n_module_epochs: int = DEFAULT_N_EPOCHS,
    module_lr: float = 5e-3,
):
    from src.methods.rfe import RfeDataArgs, RfeDataMethod

    return RfeDataMethod, RfeDataArgs(
        buffer_size=buffer_size,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=DEFAULT_SEED,
        regularize_weight=regularize_weight,
        only_last_task=only_last_task,
        d_project=d_project,
        d_aux=d_aux,
        n_module_epochs=n_module_epochs,
        module_lr=module_lr,
    )


@app.command
def cls_er(
    buffer_size: int,
    distill_weight: float,
    replay_weight: float,
    stable_frequency: float,
    stable_alpha: float,
    plastic_frequency: float,
    plastic_alpha: float,
):
    from src.methods.cls_er import ClsErArgs, ClsErMethod

    return ClsErMethod, ClsErArgs(
        buffer_size=buffer_size,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=DEFAULT_SEED,
        distill_weight=distill_weight,
        replay_weight=replay_weight,
        stable_frequency=stable_frequency,
        stable_alpha=stable_alpha,
        plastic_frequency=plastic_frequency,
        plastic_alpha=plastic_alpha,
    )


@app.command
def tamil(
    buffer_size: int,
    distill_weight: float,
    replay_weight: float,
    pairwise_weight: float,
    d_code: int,
):
    from src.methods.tamil import TamilArgs, TamilMethod

    return TamilMethod, TamilArgs(
        buffer_size=buffer_size,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=DEFAULT_SEED,
        distill_weight=distill_weight,
        replay_weight=replay_weight,
        d_code=d_code,
        pairwise_weight=pairwise_weight,
    )


@app.command
def ema_tamil(
    buffer_size: int,
    distill_weight: float,
    replay_weight: float,
    pairwise_weight: float,
    d_code: int,
    ema_frequency: float,
    ema_alpha: float,
):
    from src.methods.tamil import EmaTamilArgs, EmaTamilMethod

    return EmaTamilMethod, EmaTamilArgs(
        buffer_size=buffer_size,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=DEFAULT_SEED,
        distill_weight=distill_weight,
        replay_weight=replay_weight,
        d_code=d_code,
        pairwise_weight=pairwise_weight,
        ema_frequency=ema_frequency,
        ema_alpha=ema_alpha,
    )


@app.command
def er_mkd(
    buffer_size: int,
    distill_weight: float,
    replay_weight: float,
    ema_frequency: float,
    ema_alpha: float,
    temperature: float,
):
    from src.methods.er_mkd import ErMkdArgs, ErMkdMethod

    return ErMkdMethod, ErMkdArgs(
        buffer_size=buffer_size,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=DEFAULT_SEED,
        distill_weight=distill_weight,
        replay_weight=replay_weight,
        ema_frequency=ema_frequency,
        ema_alpha=ema_alpha,
        temperature=temperature,
    )


if __name__ == '__main__':
    app.meta()
