import attrs
import torch.nn.functional as fn
import torch.optim as optim

from . import BufferArgs, BufferMethod, ImageT, TargetT, define_module


@attrs.frozen
class DerppArgs(BufferArgs):
    distill_weight: float
    replay_weight: float


@define_module
class DerppMethod(BufferMethod[DerppArgs]):
    def replay_and_distill(self):
        if self.buffer.is_empty():
            return 0.0

        samples = self.sample_collate_buffer()
        images, targets, logits = self.sync_device(
            samples['images'], samples['targets'], samples['logits']
        )
        processed_images = self.scale_fn(self.augment_fn(images))
        drifted_logits = self.tkey_to_classifier(self.extractor(processed_images))

        replay_loss = fn.cross_entropy(drifted_logits, targets)
        distill_loss = fn.mse_loss(drifted_logits, logits)
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

        return loss.item()
