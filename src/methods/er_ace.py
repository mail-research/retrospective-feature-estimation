import attrs
import einops as ei
import numpy as np
import torch
import torch.nn.functional as fn
import torch.optim as optim

from . import BufferArgs, BufferMethod, ImageT, TargetT, TrackerMixin, VectorT, define_module

ErAceArgs = BufferArgs


@define_module
class ErAceMethod(BufferMethod[ErAceArgs], TrackerMixin):
    seen_task_target_classes: set[int] = attrs.field(init=False, factory=set)

    def see_task(self, tkey: str):
        super().see_task(tkey)
        self.seen_task_target_classes.clear()

    def see_targets(self, targets: torch.Tensor):
        self.seen_task_target_classes.update(np.unique(ei.asnumpy(targets)))

    def mask_seen_classes(self, logits: VectorT) -> VectorT:
        mask = torch.zeros_like(logits)
        mask[:, self.n_seen_target_classes :] = 1
        mask[:, list(self.seen_task_target_classes)] = 1
        return logits.masked_fill(mask == 0, -torch.inf)

    def replay(self):
        if self.buffer.is_empty():
            return 0.0

        samples = self.sample_collate_buffer()
        images, targets = self.sync_device(samples['images'], samples['targets'])
        processed_images = self.scale_fn(self.augment_fn(images))
        logits = self.tkey_to_classifier(self.extractor(processed_images))
        return fn.cross_entropy(logits, targets)

    def step(
        self,
        images: ImageT,
        targets: TargetT,
        tkey: str,
        optimizer: optim.Optimizer,
    ) -> float:
        images, targets = self.sync_device(images, targets)
        processed_images = self.scale_fn(self.augment_fn(images))

        self.see_targets(targets)

        if self.fixed:
            logits = self.tkey_to_classifier(self.extractor(processed_images))
            logits = self.mask_seen_classes(logits)
        else:
            raise NotImplementedError

        loss = fn.cross_entropy(logits, targets) + self.replay()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.modify_buffer(images=images, targets=targets)

        return loss.item()
