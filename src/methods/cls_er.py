import attrs
import jaxtyping as jty
import torch
import torch.nn.functional as fn
import torch.optim as optim

from . import BufferArgs, BufferMethod, EmaCopy, ImageT, Output, TargetT, define_module


@attrs.frozen
class ClsErArgs(BufferArgs):
    distill_weight: float
    replay_weight: float
    stable_frequency: float
    stable_alpha: float
    plastic_frequency: float
    plastic_alpha: float


@define_module
class ClsErMethod(BufferMethod[ClsErArgs]):
    stable: EmaCopy = attrs.field(init=False)
    plastic: EmaCopy = attrs.field(init=False)

    def _new_ema(self, alpha: float, frequency: float):
        return EmaCopy.from_method(
            method=self,
            alpha=alpha,
            frequency=frequency,
            seed=self.args.seed,
        )

    @stable.default  # type: ignore[operator]
    def _stable_default(self):
        return self._new_ema(alpha=self.args.stable_alpha, frequency=self.args.stable_frequency)

    @plastic.default  # type: ignore[operator]
    def _plastic_ema_default(self):
        return self._new_ema(alpha=self.args.plastic_alpha, frequency=self.args.plastic_frequency)

    def add_task(self, tkey: str, d_output: int):
        super().add_task(tkey, d_output)
        self.stable.sync_method(self)
        self.plastic.sync_method(self)

    def replay_and_distill(self):
        if self.buffer.is_empty():
            return 0.0

        samples = self.sample_collate_buffer()
        images, targets = self.sync_device(samples['images'], samples['targets'])
        processed_images = self.scale_fn(self.augment_fn(images))
        drifted_logits = self.tkey_to_classifier(self.extractor(processed_images))

        stable_logits = self.stable.tkey_to_classifier(self.stable.extractor(processed_images))
        plastic_logits = self.plastic.tkey_to_classifier(self.plastic.extractor(processed_images))
        stable_probs = fn.softmax(stable_logits, dim=-1)
        plastic_probs = fn.softmax(plastic_logits, dim=-1)
        ema_logits = torch.where(stable_probs > plastic_probs, stable_logits, plastic_logits)

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
        images, targets = self.sync_device(images, targets)
        processed_images = self.scale_fn(self.augment_fn(images))

        if self.fixed:
            logits = self.tkey_to_classifier(self.extractor(processed_images))
        else:
            raise NotImplementedError

        loss = fn.cross_entropy(logits, targets) + self.replay_and_distill()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.modify_buffer(images=images, targets=targets, logits=logits)
        self.stable.step(self)
        self.plastic.step(self)

        return loss.item()

    @torch.inference_mode()
    def infer_logits(self, images: ImageT, tkey: str | None = None) -> Output:
        images = self.sync_device(images)
        processed_images = self.scale_fn(images)

        features = self.stable.extractor(processed_images)
        if tkey is not None:
            logits = self.stable.tkey_to_classifier.module[tkey](features)
        else:
            logits = self.stable.tkey_to_classifier(features)
        return Output(logits=logits)
