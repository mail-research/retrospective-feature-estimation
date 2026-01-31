import torch.nn.functional as fn
import torch.optim as optim

from . import BufferArgs, ImageT, ReservoirBufferMethod, TargetT

ErArgs = BufferArgs


class ErMethod(ReservoirBufferMethod[ErArgs]):
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

        if self.fixed:
            logits = self.tkey_to_classifier(self.extractor(processed_images))
        else:
            logits = self.tkey_to_classifier[tkey](self.extractor(processed_images))
            targets = self.retarget(targets, tkey=tkey)

        loss = fn.cross_entropy(logits, targets) + self.replay()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.modify_buffer(images=images, targets=targets)

        return loss.item()
