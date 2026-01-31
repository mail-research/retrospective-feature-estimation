from itertools import chain

import attrs
import torch.nn.functional as fn
import torch.optim as optim

from src.methods import ImageT, Method, TargetT, TaskStartMixin


@attrs.frozen
class BasicArgs:
    reset_on_new_task: bool


class BasicMethod(Method[BasicArgs], TaskStartMixin):
    def start_task(self, *_, **__):
        if not self.args.reset_on_new_task:
            return
        for module in chain(self.extractor.modules(), self.tkey_to_classifier.modules()):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()  # type: ignore[operator]

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

        loss = fn.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
