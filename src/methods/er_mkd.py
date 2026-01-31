from copy import deepcopy

import attrs
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim

from . import BufferArgs, BufferMethod, EmaCopy, ImageT, Output, TargetT, VectorT, define_module


@attrs.frozen
class ErMkdArgs(BufferArgs):
    distill_weight: float
    replay_weight: float
    ema_frequency: float
    ema_alpha: float
    temperature: float


def kl_div(logits: VectorT, ema_logits: VectorT, temperature: float = 1.0):
    logits_log_probs = fn.log_softmax(logits / temperature, dim=-1)
    ema_probs = fn.softmax(ema_logits / temperature, dim=-1)
    return fn.kl_div(logits_log_probs, ema_probs, reduction='batchmean')


def avg_model(model1: nn.Module, model2: nn.Module) -> nn.Module:
    new_model = deepcopy(model1)
    for p, ep in zip(new_model.parameters(), model2.parameters()):
        p.data.copy_((p.data + ep.data) / 2)
    return new_model


@define_module
class ErMkdMethod(BufferMethod[ErMkdArgs]):
    ema: EmaCopy = attrs.field(init=False)

    @ema.default  # type: ignore[operator]
    def _ema_default(self):
        return EmaCopy.from_method(
            method=self,
            alpha=self.args.ema_alpha,
            frequency=self.args.ema_frequency,
            seed=self.args.seed,
        )

    @property
    def avg_extractor(self):
        return avg_model(self.extractor, self.ema.extractor)

    @property
    def avg_tkey_to_classifier(self):
        return avg_model(self.tkey_to_classifier, self.ema.tkey_to_classifier)

    def add_task(self, tkey: str, d_output: int):
        super().add_task(tkey, d_output)
        self.ema.sync_method(self)

    def replay_and_distill(self):
        if self.buffer.is_empty():
            return 0.0

        samples = self.sample_collate_buffer()
        images, targets = self.sync_device(samples['images'], samples['targets'])
        processed_images = self.scale_fn(self.augment_fn(images))
        drifed_logits = self.tkey_to_classifier(self.extractor(processed_images))
        ema_logits = self.ema.tkey_to_classifier(self.ema.extractor(processed_images))

        replay_loss = fn.cross_entropy(drifed_logits, targets)
        distill_loss = kl_div(drifed_logits, ema_logits, temperature=self.args.temperature)
        return self.args.replay_weight * replay_loss + self.args.distill_weight * distill_loss

    def step(
        self,
        images: ImageT,
        targets: TargetT,
        tkey: str,
        optimizer: optim.Optimizer,
    ) -> float:
        images, targets = self.sync_device(images, targets)
        processed_images = self.scale_fn(self.augment_fn(images))

        if self.fixed:
            logits = self.tkey_to_classifier(self.extractor(processed_images))
            ema_logits = self.ema.tkey_to_classifier(self.ema.extractor(processed_images))
        else:
            raise NotImplementedError

        ce_loss = fn.cross_entropy(logits, targets)
        mkd_loss = kl_div(logits, ema_logits, temperature=self.args.temperature)
        mkd_weight = self.args.distill_weight / self.args.replay_weight  # keep the ratio
        loss = ce_loss + mkd_weight * mkd_loss + self.replay_and_distill()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.modify_buffer(images=images, targets=targets)
        self.ema.step(self)

        return loss.item()

    @torch.inference_mode()
    def infer_logits(self, images: ImageT, tkey: str | None = None) -> Output:
        images = self.sync_device(images)
        processed_images = self.scale_fn(images)
        if tkey is not None:
            logits = self.avg_tkey_to_classifier[tkey](self.avg_extractor(processed_images))
        else:
            logits = self.avg_tkey_to_classifier(self.avg_extractor(processed_images))
        return Output(logits=logits)
