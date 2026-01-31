import os
from collections import defaultdict
from enum import Enum
from functools import partial
from typing import Any, Callable, Self

import attrs
import einops as ei
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from loguru import logger
from tqdm import tqdm

from src.datasets import Task
from src.methods import Method, Output, TaskEndMixin, TaskStartMixin

_tqdm = partial(tqdm, ncols=0, leave=False)


def process_outputs(outputs: list[Output]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    losses = np.asarray([output.loss for output in outputs])
    logits = ei.pack([ei.asnumpy(output.logits) for output in outputs], '* d')[0]
    targets = ei.pack([ei.asnumpy(output.targets) for output in outputs], '*')[0]
    return losses, logits, targets


class Setting(str, Enum):
    TASK_IL = 'task_il'
    CLASS_IL = 'class_il'


def make_loader(dataset: data_utils.Dataset, batch_size: int, training: bool = True, seed: int = 0):
    if training:
        shuffle, persistent_workers = True, True
    else:
        shuffle, persistent_workers = False, False

    return data_utils.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=min(os.cpu_count() or 4, 4),
        persistent_workers=persistent_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )


def check_lr_change(optimizer: optim.Optimizer, last_lrs: list[float] | None = None):
    current_lrs = [group['lr'] for group in optimizer.param_groups]
    if last_lrs is not None and current_lrs != last_lrs:
        logger.info(f'[Optimizer] \t Learning rate changed from {last_lrs} to {current_lrs}')
        return current_lrs
    return current_lrs


@attrs.frozen
class EvaluationArgs:
    batch_size: int
    setting: Setting
    seed: int


@attrs.frozen
class TrainingArgs(EvaluationArgs):
    lr: float
    n_epochs: int
    validation_ratio: float
    optimizer_kwargs: dict[str, Any] = attrs.field(factory=dict)
    scheduler_kwargs: dict[str, Any] = attrs.field(factory=dict)

    @property
    def ratios(self):
        if self.validation_ratio == 0.0:
            raise ValueError
        return [1 - self.validation_ratio, self.validation_ratio]

    @property
    def eval(self):
        return EvaluationArgs(batch_size=self.batch_size, seed=self.seed, setting=self.setting)


MetricDict = dict[str, float]


class MetricSeqDict(defaultdict[str, list[float]]):
    def __init__(self):
        super().__init__(list)

    def append(self, **key_to_value: float):
        for name, value in key_to_value.items():
            self[name].append(value)

    @property
    def last(self) -> MetricDict:
        return {key: values[-1] for key, values in self.items() if values}

    @property
    def summary(self) -> MetricDict:
        return {key: sum(values) / len(values) for key, values in self.items() if values}

    @classmethod
    def inflate(cls, metric_dict: MetricDict):
        new = cls()
        for key, value in metric_dict.items():
            new[key].append(value)
        return new

    @classmethod
    def concat(cls, *metric_seq_dicts: Self):
        new = cls()
        for metric_seq_dict in metric_seq_dicts:
            for key, values in metric_seq_dict.items():
                new[key].extend(values)
        return new


def stringify_metrics(**key_to_value: float) -> str:
    return ' | '.join(f'{key}={value:6.4f}' for key, value in key_to_value.items())


def stringify_epoch(epoch_idx: int, n_epochs: int):
    return str(epoch_idx + 1).zfill(len(str(n_epochs)))


def decide_inference_tkey(tkey: str, setting: Setting) -> str | None:
    match setting:
        case Setting.TASK_IL:
            return tkey
        case Setting.CLASS_IL:
            return None
        case _:
            raise ValueError


def train_task(
    args: TrainingArgs,
    method: Method,
    training_set: data_utils.Dataset,
    validation_set: data_utils.Dataset,
    tkey: str,
    metric_name_to_fn: dict[str, Callable[[np.ndarray, np.ndarray], float]],
):
    _make_loader = partial(make_loader, batch_size=args.batch_size, seed=args.seed)
    training_loader = _make_loader(training_set, training=True)
    validation_loader = _make_loader(validation_set, training=False)

    optimizer, scheduler = method.make_optimizer_with_scheduling(
        lr=args.lr,
        optim_kwargs=args.optimizer_kwargs,
        scheduler_kwargs=args.scheduler_kwargs,
    )

    if isinstance(method, TaskStartMixin):
        start_result = method.start_task(tkey, training_set, validation_set)
    else:
        start_result = None

    metric_name_to_values = MetricSeqDict()
    lrs = check_lr_change(optimizer)

    for epoch_idx in _tqdm(range(args.n_epochs), desc=f'Learning task {tkey}'):
        method.train()
        training_losses = [
            method.step(images, targets, tkey=tkey, optimizer=optimizer)
            for images, targets in _tqdm(training_loader, desc='Training')
        ]
        method.eval()
        validation_outputs = [
            method.infer(images, targets, tkey=decide_inference_tkey(tkey, args.setting))
            for images, targets in _tqdm(validation_loader, desc='Validating')
        ]

        validation_losses, logits, targets = process_outputs(validation_outputs)
        training_loss = np.mean(training_losses).item()
        validation_loss = np.mean(validation_losses).item()

        scheduler.step(validation_loss)
        lrs = check_lr_change(optimizer, last_lrs=lrs)

        metric_name_to_values.append(
            training_loss=training_loss,
            validation_loss=validation_loss,
            **{name: fn(logits, targets) for name, fn in metric_name_to_fn.items()},
        )
        logger.info(
            f'[Epoch {stringify_epoch(epoch_idx, n_epochs=args.n_epochs)}]'
            f'\t{stringify_metrics(**metric_name_to_values.last)}'
        )

    if isinstance(method, TaskEndMixin):
        end_result = method.end_task(tkey, training_set, validation_set)
    else:
        end_result = None

    return metric_name_to_values, (start_result, end_result)


def eval_learned_tasks(
    args: EvaluationArgs,
    method: Method,
    tkey_to_dataset: dict[str, data_utils.Dataset],
    metric_name_to_fn: dict[str, Callable[[np.ndarray, np.ndarray], float]],
):
    tkey_to_metric_name_to_value = defaultdict[str, MetricDict](MetricDict)
    for tkey, dataset in tkey_to_dataset.items():
        loader = make_loader(dataset, batch_size=args.batch_size, training=False, seed=args.seed)
        method.eval()
        outputs = [
            method.infer(images, targets, tkey=decide_inference_tkey(tkey, args.setting))
            for images, targets in _tqdm(loader, desc=f'Testing {tkey}')
        ]
        losses, logits, targets = process_outputs(outputs)
        tkey_to_metric_name_to_value[tkey].update(
            loss=np.mean(losses).item(),
            **{name: fn(logits, targets) for name, fn in metric_name_to_fn.items()},
        )
        logger.info(f'[Eval {tkey}]\t{stringify_metrics(**tkey_to_metric_name_to_value[tkey])}')

    return tkey_to_metric_name_to_value


@attrs.define
class TaskResult:
    training: MetricSeqDict
    validation: dict[str, MetricDict]
    test: dict[str, MetricDict]
    extra: Any | None = None

    @property
    def validation_agg(self):
        return MetricSeqDict.concat(*map(MetricSeqDict.inflate, self.validation.values()))

    @property
    def test_agg(self):
        return MetricSeqDict.concat(*map(MetricSeqDict.inflate, self.test.values()))

    @property
    def validation_summary(self):
        return self.validation_agg.summary

    @property
    def test_summary(self):
        return self.test_agg.summary


def learn(
    method: Method,
    tkey_to_task: dict[str, Task],
    args: TrainingArgs,
    metric_name_to_fn: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    expanding: bool = False,
):
    if not expanding:  # add all heads at once
        for tkey, task in tkey_to_task.items():
            method.add_task(tkey=tkey, d_output=len(task.target_classes))
        method.fixed = True  # if the number of classifier heads is fixed

    tkey_to_validation_set, tkey_to_test_set = {}, {}
    for tkey, task in tkey_to_task.items():
        if expanding:
            method.add_task(tkey=tkey, d_output=len(task.target_classes))

        method.see_task(tkey=tkey)

        training_set, validation_set = data_utils.random_split(
            dataset=task.training_set,
            lengths=args.ratios,
            generator=torch.Generator().manual_seed(args.seed),
        )
        logger.info('[TRAINING]')
        training_metric_name_to_values, extra_results = train_task(
            args=args,
            method=method,
            training_set=training_set,
            validation_set=validation_set,
            tkey=tkey,
            metric_name_to_fn=metric_name_to_fn,
        )

        logger.info('[VALIDATION]')
        tkey_to_validation_set[tkey] = validation_set
        tkey_to_validation_metric_name_to_value = eval_learned_tasks(
            args=args.eval,
            method=method,
            tkey_to_dataset=tkey_to_validation_set,  # type: ignore[arg-type]
            metric_name_to_fn=metric_name_to_fn,
        )

        logger.info('[TEST]')
        tkey_to_test_set[tkey] = task.test_set
        tkey_to_test_metric_name_to_value = eval_learned_tasks(
            args=args.eval,
            method=method,
            tkey_to_dataset=tkey_to_test_set,
            metric_name_to_fn=metric_name_to_fn,
        )

        task_result = TaskResult(
            training=training_metric_name_to_values,
            validation=tkey_to_validation_metric_name_to_value,
            test=tkey_to_test_metric_name_to_value,
            extra=extra_results,
        )
        logger.info('[SUMMARY]')
        logger.info(f'[Validation]\t{stringify_metrics(**task_result.validation_summary)}')
        logger.info(f'[Test]\t{stringify_metrics(**task_result.test_summary)}')
        yield tkey, task_result


learn_fixed = partial(learn, expanding=False)
learn_expanding = partial(learn, expanding=True)


def learn_joint(
    method: Method,
    tkey_to_task: dict[str, Task],
    args: TrainingArgs,
    metric_name_to_fn: dict[str, Callable[[np.ndarray, np.ndarray], float]],
):
    for tkey, task in tkey_to_task.items():
        method.add_task(tkey=tkey, d_output=len(task.target_classes))
        method.see_task(tkey=tkey)
    method.fixed = True

    training_sets = []
    tkey_to_validation_set, tkey_to_test_set = {}, {}
    for tkey, task in tkey_to_task.items():
        training_set, validation_set = data_utils.random_split(
            dataset=task.training_set,
            lengths=args.ratios,
            generator=torch.Generator().manual_seed(args.seed),
        )
        training_sets.append(training_set)
        tkey_to_validation_set[tkey] = validation_set
        tkey_to_test_set[tkey] = task.test_set

    logger.info('[TRAINING]')
    training_metric_name_to_values, extra_results = train_task(
        args=attrs.evolve(args, setting=Setting.CLASS_IL),
        method=method,
        training_set=data_utils.ConcatDataset(training_sets),
        validation_set=data_utils.ConcatDataset(list(tkey_to_validation_set.values())),
        tkey='all',
        metric_name_to_fn=metric_name_to_fn,
    )

    logger.info('[VALIDATION]')
    tkey_to_validation_metric_name_to_value = eval_learned_tasks(
        args=args.eval,
        method=method,
        tkey_to_dataset=tkey_to_validation_set,  # type: ignore[arg-type]
        metric_name_to_fn=metric_name_to_fn,
    )
    logger.info('[TEST]')
    tkey_to_test_metric_name_to_value = eval_learned_tasks(
        args=args.eval,
        method=method,
        tkey_to_dataset=tkey_to_test_set,
        metric_name_to_fn=metric_name_to_fn,
    )

    task_result = TaskResult(
        training=training_metric_name_to_values,
        validation=tkey_to_validation_metric_name_to_value,
        test=tkey_to_test_metric_name_to_value,
        extra=extra_results,
    )
    logger.info('[SUMMARY]')
    logger.info(f'[Validation]\t{stringify_metrics(**task_result.validation_summary)}')
    logger.info(f'[Test]\t{stringify_metrics(**task_result.test_summary)}')
    yield tkey, task_result
