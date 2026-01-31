from typing import TypeVar

import attrs
import einops as ei
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim

from . import (
    BufferArgs,
    BufferMethod,
    EmaCopy,
    ImageT,
    Output,
    TargetT,
    TrackerMixin,
    VectorT,
    define_module,
)


class Attention(nn.Module):
    def __init__(self, d_input: int, d_code: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_input, d_code), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(d_code, d_input), nn.Sigmoid())

    def forward(self, features: torch.Tensor):
        return self.decoder(self.encoder(features))


class AttentionModuleDict(nn.ModuleDict):
    def forward(self, features: VectorT):
        tasks_attentions = ei.pack(
            [attentions(features) for attentions in self.values()],
            pattern='* b d',
        )[0]
        tasks_features = ei.repeat(features, 'b d -> n_tasks b d', n_tasks=len(self))

        tasks_errors = ei.reduce(
            fn.mse_loss(tasks_attentions, tasks_features, reduction='none'),
            pattern='n_tasks b d-> n_tasks b',
            reduction=torch.mean,
        )
        selected_task_idxs = torch.argmin(tasks_errors, dim=0)

        selected_attentions = tasks_attentions[selected_task_idxs, torch.arange(len(features)), :]
        return selected_attentions


@attrs.frozen
class TamilArgs(BufferArgs):
    d_code: int
    pairwise_weight: float
    replay_weight: float
    distill_weight: float


TamilArgsT = TypeVar('TamilArgsT', bound=TamilArgs)


@define_module
class TamilMethod(BufferMethod[TamilArgsT], TrackerMixin):
    tkey_to_attention: AttentionModuleDict = attrs.field(init=False, factory=AttentionModuleDict)

    def see_task(self, tkey: str):
        super().see_task(tkey)
        self.tkey_to_attention[tkey] = Attention(self.extractor.d_output, self.args.d_code)
        self.sync_device(self.tkey_to_attention)

    def replay_and_distill(self):
        if self.buffer.is_empty():
            return 0.0

        samples = self.sample_collate_buffer()
        images, targets, logits = self.sync_device(
            samples['images'], samples['targets'], samples['logits']
        )
        processed_images = self.scale_fn(self.augment_fn(images))
        drifted_features = self.extractor(processed_images)
        attention = self.tkey_to_attention(drifted_features)
        drifted_logits = self.tkey_to_classifier(drifted_features * attention)

        replay_loss = fn.cross_entropy(drifted_logits, targets)
        distill_loss = fn.mse_loss(drifted_logits, logits)
        return self.args.replay_weight * replay_loss + self.args.distill_weight * distill_loss

    def maximize_attention_discrepancy(self, features: VectorT, attentions: VectorT):
        if self.n_learned_tasks == 0:
            return 0.0

        with torch.no_grad():
            learned_tasks_attentions = ei.pack(
                [self.tkey_to_attention[key](features) for key in self.learned_tkeys],
                pattern='* b d',
            )[0]

        pairwise_loss = -fn.l1_loss(
            ei.repeat(fn.softmax(attentions, dim=-1), 'b d -> t b d', t=self.n_learned_tasks),
            fn.softmax(learned_tasks_attentions, dim=-1),
        )
        return self.args.pairwise_weight * pairwise_loss

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
        attentions = self.tkey_to_attention[tkey](features)

        if self.fixed:
            logits = self.tkey_to_classifier(attentions * features)
        else:
            raise NotImplementedError

        loss = (
            fn.cross_entropy(logits, targets)
            + self.replay_and_distill()
            + self.maximize_attention_discrepancy(features, attentions)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tkeys = [tkey] * len(images)
        self.modify_buffer(images=images, targets=targets, logits=logits, tkeys=tkeys)

        return loss.item()

    @torch.inference_mode()
    def infer_logits(self, images: ImageT, tkey: str | None = None) -> Output:
        images = self.sync_device(images)
        processed_images = self.scale_fn(images)
        features = self.extractor(processed_images)
        if tkey is not None:
            attentions = self.tkey_to_attention[tkey](features)
            logits = self.tkey_to_classifier[tkey](features * attentions)
        else:
            attentions = self.tkey_to_attention(features)
            logits = self.tkey_to_classifier(features * attentions)
        return Output(logits=logits)


@attrs.frozen
class EmaTamilArgs(TamilArgs):
    ema_frequency: float
    ema_alpha: float


@define_module
class EmaTamilMethod(TamilMethod[EmaTamilArgs]):
    ema: EmaCopy = attrs.field(init=False)

    @ema.default  # type: ignore[operator]
    def _ema_default(self):
        return EmaCopy.from_method(
            method=self,
            alpha=self.args.ema_alpha,
            frequency=self.args.ema_frequency,
            seed=self.args.seed,
        )

    def add_task(self, tkey: str, d_output: int):
        super().add_task(tkey, d_output)
        self.ema.sync_method(self)

    def replay_and_distill(self):
        if self.buffer.is_empty():
            return 0.0

        samples = self.sample_collate_buffer()
        images, targets = self.sync_device(samples['images'], samples['targets'])
        processed_images = self.scale_fn(self.augment_fn(images))

        drifted_features = self.extractor(processed_images)
        ema_features = self.ema.extractor(processed_images)

        with torch.no_grad():
            drifted_attentions = self.tkey_to_attention(drifted_features)
            all_tasks_attentions = ei.pack(
                [attention(ema_features) for attention in self.tkey_to_attention.values()],
                pattern='* b d',
            )[0]
            tkey_to_idx = {tkey: idx for idx, tkey in enumerate(self.tkey_to_attention.keys())}
            selected_idxs = torch.tensor([tkey_to_idx[tkey] for tkey in samples['tkeys']])
            ema_attentions = all_tasks_attentions[selected_idxs, torch.arange(len(ema_features)), :]

        drifted_logits = self.tkey_to_classifier(drifted_features * drifted_attentions)
        ema_logits = self.ema.tkey_to_classifier(ema_features * ema_attentions)

        replay_loss = fn.cross_entropy(drifted_logits, targets)
        distill_loss = fn.mse_loss(drifted_logits, ema_logits)
        return self.args.replay_weight * replay_loss + self.args.distill_weight * distill_loss

    def step(
        self,
        images: ImageT,
        targets: TargetT,
        tkey: str,
        optimizer: optim.Optimizer,
    ) -> float:
        loss = super().step(images, targets, tkey, optimizer)
        self.ema.step(self)
        return loss
