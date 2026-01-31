from copy import deepcopy
from functools import partial
from typing import Iterable, TypeVar

import attrs
import einops as ei
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from einops.layers.torch import Rearrange, Reduce
from loguru import logger
from tqdm import tqdm

from src.methods import (
    BufferArgs,
    DatasetT,
    ImageT,
    Output,
    ReservoirBufferMethod,
    TargetT,
    TaskEndMixin,
    TrackerMixin,
    VectorT,
    define_module,
)
from src.modules.net import FeatureExtractor
from src.utils.misc import freeze
from src.utils.training import (
    MetricSeqDict,
    check_lr_change,
    make_loader,
    stringify_epoch,
    stringify_metrics,
)

_tqdm = partial(tqdm, ncols=0, leave=False)


class AuxFeatureExtractor(nn.Module):
    def __init__(self, d_output: int):
        super().__init__()
        Conv2d = partial(nn.Conv2d, kernel_size=3, stride=2, padding=1)
        self.net = nn.Sequential(
            Reduce('b c (h dh) (w dw) -> b c h w', c=3, h=16, w=16, reduction='mean'),
            Conv2d(in_channels=3, out_channels=d_output // 2),  # (d/2)x8x8
            nn.ReLU(),
            nn.MaxPool2d(2),  # (d/2)x4x4
            Conv2d(in_channels=d_output // 2, out_channels=d_output),  # dx2x2
            nn.ReLU(),
            nn.MaxPool2d(2),  # dx1x1
            Rearrange('b c 1 1 -> b c', c=d_output),  # d
        )

    def forward(self, inputs: torch.Tensor):
        return self.net(inputs)


class Retrospector(nn.Module):
    def __init__(self, d_feature: int, d_aux: int, d_projection: int):
        super().__init__()
        self.aux_project = nn.Linear(d_aux, d_projection)
        self.main_project = nn.Linear(d_feature, d_projection)

        self.aux_scale = nn.Linear(d_aux, d_feature)

        self.aux_score = nn.Sequential(nn.Linear(d_projection, d_feature), nn.Sigmoid())
        self.main_score = nn.Sequential(nn.Linear(d_projection, d_feature), nn.Sigmoid())

    def forward(self, features: torch.Tensor, aux_features: torch.Tensor):
        projected_aux_features = self.aux_project(aux_features)
        projected_main_features = self.main_project(features)

        similarity = projected_aux_features * projected_main_features

        aux_score = self.aux_score(similarity)
        main_score = self.main_score(similarity)

        return fn.relu(aux_score * self.aux_scale(aux_features) + main_score * features)


@attrs.frozen
class RfeArgs(BufferArgs):
    regularize_weight: float | None
    d_project: int
    d_aux: int
    n_module_epochs: int
    module_lr: float

    buffer_size: int = attrs.field(init=False, default=0)


RfeArgsT = TypeVar('RfeArgsT', bound=RfeArgs)


@define_module
class RfeMethod(ReservoirBufferMethod[RfeArgsT], TaskEndMixin, TrackerMixin):
    last_extractor: FeatureExtractor | None = attrs.field(init=False, default=None)
    last_tkey_to_classifier: nn.Module | None = attrs.field(init=False, default=None)
    tkey_to_aux_extractor: nn.ModuleDict = attrs.field(init=False, factory=nn.ModuleDict)
    tkey_to_retrospector: nn.ModuleDict = attrs.field(init=False, factory=nn.ModuleDict)

    def loss_aux_extractor(self, processed_images: ImageT, features: VectorT, adapter: nn.Module):
        aux_features = adapter(self.tkey_to_aux_extractor[self.current_tkey](processed_images))
        return fn.mse_loss(aux_features, features)

    def loss_restrospector(self, processed_images: ImageT, features: VectorT):
        if self.last_extractor is None or self.last_tkey is None:
            return self.sync_device(torch.tensor(0.0))

        with torch.no_grad():
            past_features = self.last_extractor(processed_images)
            past_aux_features = self.tkey_to_aux_extractor[self.last_tkey](processed_images)

        rectified_features = self.tkey_to_retrospector[self.last_tkey](features, past_aux_features)
        return fn.mse_loss(rectified_features, past_features)

    def train_modules(self, training_set: DatasetT, validation_set: DatasetT):
        aux_extractor = AuxFeatureExtractor(self.args.d_aux)
        self.tkey_to_aux_extractor[self.current_tkey] = self.sync_device(aux_extractor)

        adapter = nn.Linear(self.args.d_aux, self.extractor.d_output)
        self.sync_device(adapter)

        modules = nn.ModuleList([self.tkey_to_aux_extractor, adapter])

        if self.last_tkey is not None:
            last_retrospector = Retrospector(
                d_feature=self.extractor.d_output,
                d_aux=self.args.d_aux,
                d_projection=self.args.d_project,
            )
            self.tkey_to_retrospector[self.last_tkey] = self.sync_device(last_retrospector)
            modules.append(self.tkey_to_retrospector)

        optimizer, scheduler = self.make_optimizer_with_scheduling(
            lr=self.args.module_lr,
            parameters=modules.parameters(),
            scheduler_kwargs={'patience': 3, 'factor': 0.1},
        )

        @torch.no_grad()
        def validate(images: torch.Tensor):
            images = self.sync_device(images)
            processed_images = self.scale_fn(images)
            features = self.extractor(processed_images)

            return (
                self.loss_aux_extractor(processed_images, features, adapter=adapter).item(),
                self.loss_restrospector(processed_images, features).item(),
            )

        def train(images):  # combinining unrelated components for faster training
            images = self.sync_device(images)
            processed_images = self.scale_fn(self.augment_fn(images))
            with torch.no_grad():
                features = self.extractor(processed_images)

            loss_aux_extractor = self.loss_aux_extractor(processed_images, features, adapter)
            loss_restrospector = self.loss_restrospector(processed_images, features)
            loss = loss_aux_extractor + loss_restrospector
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss_aux_extractor.item(), loss_restrospector.item()

        _make_loader = partial(make_loader, batch_size=self.args.batch_size)
        training_loader = _make_loader(training_set, training=True)
        validation_loader = _make_loader(validation_set, training=False)

        metric_name_to_values = MetricSeqDict()
        last_lrs = check_lr_change(optimizer)
        for epoch_idx in _tqdm(range(self.args.n_module_epochs), desc='Training module'):
            modules.train()
            paired_training_losses = [train(images) for images, _ in _tqdm(training_loader)]
            aux_extractor_training_losses = [loss for loss, _ in paired_training_losses]
            retrospector_training_losses = [loss for _, loss in paired_training_losses]

            modules.eval()
            paired_validation_losses = [validate(images) for images, _ in _tqdm(validation_loader)]
            aux_extractor_validation_losses = [loss for loss, _ in paired_validation_losses]
            retrospector_validation_losses = [loss for _, loss in paired_validation_losses]

            metric_name_to_values.append(
                aux_extractor_training_loss=np.mean(aux_extractor_training_losses).item(),
                retrospector_training_loss=np.mean(retrospector_training_losses).item(),
                aux_extractor_validation_loss=np.mean(aux_extractor_validation_losses).item(),
                retrospector_validation_loss=np.mean(retrospector_validation_losses).item(),
            )

            scheduler.step(
                np.mean(aux_extractor_validation_losses).item()
                + np.mean(retrospector_validation_losses).item()
            )
            last_lrs = check_lr_change(optimizer, last_lrs)
            logger.info(
                f'[Modules {stringify_epoch(epoch_idx, self.args.n_module_epochs)}] '
                f'{stringify_metrics(**metric_name_to_values.last)}'
            )
        return metric_name_to_values

    def end_task(self, tkey: str, training_set: DatasetT, validation_set: DatasetT):
        self.eval()
        freeze(self.tkey_to_classifier[tkey])
        result = self.train_modules(training_set=training_set, validation_set=validation_set)
        self.last_extractor = deepcopy(self.extractor)
        self.last_tkey_to_classifier = deepcopy(self.tkey_to_classifier)
        return result

    def chain_retrospect(self, images: ImageT, features: VectorT, ordered_tkeys: Iterable[str]):
        for tkey in ordered_tkeys:
            with torch.no_grad():
                aux_features = self.tkey_to_aux_extractor[tkey](images)
            features = self.tkey_to_retrospector[tkey](features, aux_features)
            yield tkey, features

    @torch.inference_mode()
    def infer_logits(self, images: ImageT, tkey: str | None = None):
        images = self.sync_device(images)
        processed_images = self.scale_fn(images)

        features = self.extractor(processed_images)
        tkeys = list(reversed(self.tkey_to_retrospector.keys()))

        def mask(logits: torch.Tensor, start: int, end: int | None = None):
            logits[:, :start] = -torch.inf
            if end is not None:
                logits[:, end:] = -torch.inf
            return logits

        if tkey is None:
            start, end = self.n_learned_target_classes, self.n_seen_target_classes
            all_logits = [mask(self.tkey_to_classifier(features), start=start, end=end)]

            n_retrospector_target_classes = sum(self.tkey_to_d_output[key] for key in tkeys)
            if n_retrospector_target_classes < start:
                # which means the latest retrospector has not been learned yet
                start, end = n_retrospector_target_classes, self.n_learned_target_classes
                all_logits.append(mask(self.tkey_to_classifier(features), start=start, end=end))

            for key, features in self.chain_retrospect(processed_images, features, tkeys):
                n_task_target_classes = self.tkey_to_d_output[key]
                start, end = start - n_task_target_classes, start
                all_logits.append(mask(self.tkey_to_classifier(features), start=start, end=end))

            probs = ei.reduce(
                fn.softmax(ei.pack(all_logits, pattern='* b d')[0], dim=-1),
                pattern='t b d -> b d',
                reduction=torch.mean,
            )
        else:
            if len(tkeys) != 0 and tkey in tkeys:
                for key, features in self.chain_retrospect(processed_images, features, tkeys):
                    if tkey == key:
                        break
            probs = fn.softmax(self.tkey_to_classifier[tkey](features), dim=-1)
        return Output(logits=probs)

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
        loss = fn.nll_loss(outputs.logits.log(), targets).item()
        return attrs.evolve(outputs, targets=targets, loss=loss)

    def regularize(self, processed_images: ImageT, features: VectorT):
        if self.args.regularize_weight is None or self.last_extractor is None:
            return 0.0

        with torch.no_grad():
            past_features = self.last_extractor(processed_images)
        return fn.mse_loss(features, past_features) * self.args.regularize_weight

    def step(
        self,
        images: ImageT,
        targets: TargetT,
        tkey: str,
        optimizer: optim.Optimizer,
    ) -> float:
        images, targets = self.sync_device(images, targets)
        processed_images = self.scale_fn(self.augment_fn(images))
        features = self.extractor(processed_images)

        if self.fixed:
            logits = self.tkey_to_classifier(features)
        else:
            logits = self.tkey_to_classifier[tkey](features)
            targets = self.retarget(targets, tkey=tkey)

        loss = fn.cross_entropy(
            logits[:, self.n_learned_target_classes :],
            targets - self.n_learned_target_classes,
        ) + self.regularize(processed_images, features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


@define_module  # end-to-end RFE without separate training for modules
class RfeE2EMethod(ReservoirBufferMethod[RfeArgsT], TaskEndMixin, TrackerMixin):
    last_extractor: FeatureExtractor | None = attrs.field(init=False, default=None)
    last_tkey_to_classifier: nn.Module | None = attrs.field(init=False, default=None)
    tkey_to_aux_extractor: nn.ModuleDict = attrs.field(init=False, factory=nn.ModuleDict)
    tkey_to_retrospector: nn.ModuleDict = attrs.field(init=False, factory=nn.ModuleDict)

    def loss_restrospector(self, processed_images: ImageT, features: VectorT):
        if self.last_extractor is None or self.last_tkey is None:
            return self.sync_device(torch.tensor(0.0))

        with torch.no_grad():
            past_features = self.last_extractor(processed_images)

        past_aux_features = self.tkey_to_aux_extractor[self.last_tkey](processed_images)
        rectified_features = self.tkey_to_retrospector[self.last_tkey](features, past_aux_features)
        return fn.mse_loss(rectified_features, past_features)

    def end_task(self, tkey: str, training_set: DatasetT, validation_set: DatasetT):
        self.eval()
        freeze(self.tkey_to_classifier[tkey])
        self.last_extractor = deepcopy(self.extractor)
        self.last_tkey_to_classifier = deepcopy(self.tkey_to_classifier)
        self.tkey_to_aux_extractor[tkey] = AuxFeatureExtractor(self.args.d_aux)
        self.tkey_to_retrospector[tkey] = Retrospector(
            d_feature=self.extractor.d_output,
            d_aux=self.args.d_aux,
            d_projection=self.args.d_project,
        )
        self.sync_device(self.tkey_to_aux_extractor[tkey], self.tkey_to_retrospector[tkey])

    def chain_retrospect(self, images: ImageT, features: VectorT, ordered_tkeys: Iterable[str]):
        for tkey in ordered_tkeys:
            with torch.no_grad():
                aux_features = self.tkey_to_aux_extractor[tkey](images)
            features = self.tkey_to_retrospector[tkey](features, aux_features)
            yield tkey, features

    @torch.inference_mode()
    def infer_logits(self, images: ImageT, tkey: str | None = None):
        images = self.sync_device(images)
        processed_images = self.scale_fn(images)

        features = self.extractor(processed_images)
        tkeys = list(reversed(self.tkey_to_retrospector.keys()))

        def mask(logits: torch.Tensor, start: int, end: int | None = None):
            logits[:, :start] = -torch.inf
            if end is not None:
                logits[:, end:] = -torch.inf
            return logits

        if tkey is None:
            start, end = self.n_learned_target_classes, self.n_seen_target_classes
            all_logits = [mask(self.tkey_to_classifier(features), start=start, end=end)]

            n_retrospector_target_classes = sum(self.tkey_to_d_output[key] for key in tkeys)
            if n_retrospector_target_classes < start:
                # which means the latest retrospector has not been learned yet
                start, end = n_retrospector_target_classes, self.n_learned_target_classes
                all_logits.append(mask(self.tkey_to_classifier(features), start=start, end=end))

            for key, features in self.chain_retrospect(processed_images, features, tkeys):
                n_task_target_classes = self.tkey_to_d_output[key]
                start, end = start - n_task_target_classes, start
                all_logits.append(mask(self.tkey_to_classifier(features), start=start, end=end))

            probs = ei.reduce(
                fn.softmax(ei.pack(all_logits, pattern='* b d')[0], dim=-1),
                pattern='t b d -> b d',
                reduction=torch.mean,
            )
        else:
            if len(tkeys) != 0 and tkey in tkeys:
                for key, features in self.chain_retrospect(processed_images, features, tkeys):
                    if tkey == key:
                        break
            probs = fn.softmax(self.tkey_to_classifier[tkey](features), dim=-1)
        return Output(logits=probs)

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
        loss = fn.nll_loss(outputs.logits.log(), targets).item()
        return attrs.evolve(outputs, targets=targets, loss=loss)

    def regularize(self, processed_images: ImageT, features: VectorT):
        if self.args.regularize_weight is None or self.last_extractor is None:
            return 0.0

        return self.loss_restrospector(processed_images, features) * self.args.regularize_weight

    def step(
        self,
        images: ImageT,
        targets: TargetT,
        tkey: str,
        optimizer: optim.Optimizer,
    ) -> float:
        images, targets = self.sync_device(images, targets)
        processed_images = self.scale_fn(self.augment_fn(images))
        features = self.extractor(processed_images)

        if self.fixed:
            logits = self.tkey_to_classifier(features)
        else:
            logits = self.tkey_to_classifier[tkey](features)
            targets = self.retarget(targets, tkey=tkey)

        loss = fn.cross_entropy(
            logits[:, self.n_learned_target_classes :],
            targets - self.n_learned_target_classes,
        ) + self.regularize(processed_images, features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


@attrs.frozen
class RfeDataArgs(RfeArgs):
    buffer_size: int
    only_last_task: bool


@define_module
class RfeDataMethod(RfeMethod[RfeDataArgs]):
    @torch.no_grad()
    def load_buffer(self, tkey: str, dataset: DatasetT):
        if self.args.only_last_task:
            self.buffer.empty()

        loader = make_loader(dataset, batch_size=self.args.batch_size, training=False)
        for images, _ in _tqdm(loader, desc='Saving'):
            images = self.sync_device(images)
            processed_images = self.scale_fn(images)
            features = self.extractor(processed_images)
            logits = self.tkey_to_classifier(features)
            tkeys = [tkey] * len(images)
            self.modify_buffer(images=images, features=features, logits=logits, tkeys=tkeys)

    def end_task(self, tkey: str, training_set: DatasetT, validation_set: DatasetT):
        result = super().end_task(tkey, training_set, validation_set)
        self.load_buffer(tkey, training_set)
        return result

    def loss_retrospector_with_buffer(self):
        if self.buffer.is_empty():
            return self.sync_device(torch.tensor(0.0))

        ordered_tkeys = list(reversed(self.learned_tkeys))

        samples = self.sample_collate_buffer()
        images, features = self.sync_device(samples['images'], samples['features'])
        tkey_to_idx = {key: idx for idx, key in enumerate(ordered_tkeys)}
        farthest_key = max(samples['tkeys'], key=lambda key: tkey_to_idx[key])
        selected_idxs = torch.tensor([tkey_to_idx[tkey] for tkey in samples['tkeys']])

        processed_images = self.scale_fn(self.augment_fn(images))
        with torch.no_grad():
            drifted_features = self.extractor(processed_images)

        all_rectified_features = []
        for key, rectified_features in self.chain_retrospect(
            processed_images, drifted_features, ordered_tkeys
        ):
            all_rectified_features.append(rectified_features)
            if key == farthest_key:
                break

        rectified_features = ei.pack(all_rectified_features, '* b d')[0][
            selected_idxs, torch.arange(len(images)), :
        ]
        return fn.mse_loss(rectified_features, features)

    def loss_restrospector(self, processed_images: ImageT, features: VectorT):
        return (
            super().loss_restrospector(processed_images, features)
            + self.loss_retrospector_with_buffer()
        )

    def regularize_with_buffer(self):
        if self.buffer.is_empty():
            return self.sync_device(torch.tensor(0.0))

        samples = self.sample_collate_buffer()
        images, features = self.sync_device(samples['images'], samples['features'])
        processed_images = self.scale_fn(self.augment_fn(images))
        drifted_features = self.extractor(processed_images)
        return fn.mse_loss(drifted_features, features) * self.args.regularize_weight  # type: ignore[operator]

    def regularize(self, processed_images: ImageT, features: VectorT):
        return super().regularize(processed_images, features) + self.regularize_with_buffer()
