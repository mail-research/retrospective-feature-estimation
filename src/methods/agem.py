from typing import Iterable, List

import einops as ei
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from tqdm import tqdm

from src.methods import BufferArgs, BufferMethod, DatasetT, ImageT, TargetT, TaskEndMixin
from src.utils.training import make_loader


def extract_grads(parameters: Iterable[nn.Parameter]):
    packed_grad, packed_shape = ei.pack([param.grad.clone() for param in parameters], '*')  # type: ignore[union-attr]
    return packed_grad, packed_shape


def write_grads(
    parameters: Iterable[nn.Parameter],
    packed_grad: torch.Tensor,
    packed_shape: List[ei.packing.Shape],
):
    grads = ei.unpack(packed_grad, packed_shape, pattern='*')
    for param, grad in zip(parameters, grads):
        param.grad = grad


def project(current_grads: torch.Tensor, buffer_grads: torch.Tensor):
    corr = torch.dot(current_grads, buffer_grads) / torch.dot(buffer_grads, buffer_grads)
    return current_grads - corr * buffer_grads


AgemArgs = BufferArgs


class AgemMethod(BufferMethod[AgemArgs], TaskEndMixin):
    def end_task(self, tkey: str, training_set: DatasetT, validation_set: DatasetT):
        loader = make_loader(training_set, batch_size=self.args.batch_size, training=False)
        for images, targets in tqdm(loader, desc='Filling buffer', ncols=0, leave=False):
            self.modify_buffer(images=images, targets=targets)

    def replay(self):
        if self.buffer.is_empty():
            return 0.0

        samples = self.sample_collate_buffer()
        images, targets = self.sync_device(samples['images'], samples['targets'])
        processed_images = self.scale_fn(self.augment_fn(images))
        logits = self.tkey_to_classifier(self.extractor(processed_images))
        return fn.cross_entropy(logits, targets)

    def project_grad(self):
        replay_loss = self.replay()
        if replay_loss == 0.0:
            return

        grads, packed_shape = extract_grads(self.parameters())
        self.zero_grad()
        replay_loss.backward()
        replay_grads, _ = extract_grads(self.parameters())

        if torch.dot(grads, replay_grads).item() < 0:
            grads = project(grads, replay_grads)

        write_grads(self.parameters(), grads, packed_shape=packed_shape)

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

        loss = fn.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        self.project_grad()
        optimizer.step()

        return loss.item()
