import zipfile
from functools import partial
from hashlib import md5
from pathlib import Path

import attrs
import einops as ei
import httpx
import jaxtyping as jty
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from loguru import logger
from PIL import Image
from tqdm import tqdm

_tqdm = partial(tqdm, ncols=0)


@attrs.define
class MetaDataset:
    name: str
    training_set: data_utils.Dataset
    test_set: data_utils.Dataset
    scale_fn: nn.Module = attrs.field(factory=nn.Identity)
    augment_fn: nn.Module = attrs.field(factory=nn.Identity)


@attrs.define
class SimpleDataset(data_utils.Dataset):
    images: jty.Float[np.ndarray, 'b h w c']
    targets: jty.Int[np.ndarray, 'b']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
        return self.images[idx], self.targets[idx]

    def __repr__(self):
        return f'{type(self).__name__}[{len(self)}]'


def download_dataset(data_url: str, root_path: Path, md5_hash: str, force: bool = False) -> Path:
    root_path.mkdir(parents=True, exist_ok=True)
    data_path = root_path / data_url.split('/')[-1]

    def is_valid(file_path: Path, md5_hash: str) -> bool:
        with open(file_path, 'rb') as file:
            file_md5 = md5(file.read()).hexdigest()
        return file_md5 == md5_hash

    if data_path.exists():
        if is_valid(data_path, md5_hash) and not force:
            logger.info(f'Dataset already exists at {data_path}')
            return data_path

    logger.info('Dataset not found or invalid, downloading...')
    with (
        httpx.stream('GET', data_url, follow_redirects=True) as response,
        data_path.open('wb') as file,
    ):
        response.raise_for_status()
        for chunk in _tqdm(response.iter_bytes(), unit=' chunks'):
            file.write(chunk)

    if not is_valid(data_path, md5_hash):
        raise RuntimeError('Failed to download dataset, md5 mismatch')

    return data_path


def make_dataset(data_path: Path):
    images, targets = [], []
    with zipfile.ZipFile(data_path) as file:
        logger.info('Loading dataset')
        image_paths = sorted(zipfile.Path(file).iterdir(), key=lambda path: path.name)
        for image_path in _tqdm(image_paths, unit=' files', leave=False):
            _, target = image_path.stem.split('.')
            with image_path.open('rb') as image_file:
                images.append(np.asarray(Image.open(image_file, formats=['WEBP'])))
            targets.append(int(target))
    all_images = ei.rearrange(ei.pack(images, '* h w c')[0], 'b h w c -> b c h w')
    all_targets = np.asarray(targets)
    all_images, all_targets = np.ascontiguousarray(all_images), np.ascontiguousarray(all_targets)
    logger.info(f'Images: {all_images.shape}, Labels: {all_targets.shape}')
    return SimpleDataset(all_images.astype(np.float32) / 255.0, all_targets.astype(np.int64))


def split_dataset(dataset: data_utils.Dataset, tkey_to_target_classes: dict[str, list[int]]):
    tkeys = tkey_to_target_classes.keys()
    target_class_to_tkey = {
        target_class: tkey for tkey in tkeys for target_class in tkey_to_target_classes[tkey]
    }

    def iter_dataset(dataset):
        for image, target in _tqdm(dataset, desc='Splitting', leave=False):
            tkey = target_class_to_tkey.get(int(target))
            if not tkey:
                raise ValueError(f'No task key found for target class {target}')
            yield tkey, np.asarray(image), int(target)

    tkey_to_images: dict[str, list[jty.Float[np.ndarray, 'h w c']]] = {tkey: [] for tkey in tkeys}
    tkey_to_targets: dict[str, list[int]] = {tkey: [] for tkey in tkeys}
    for tkey, image, target in iter_dataset(dataset):
        tkey_to_images[tkey].append(image)
        tkey_to_targets[tkey].append(target)

    return {
        tkey: SimpleDataset(images=ei.pack(images, '* c h w')[0], targets=np.asarray(targets))
        for tkey, images, targets in zip(tkeys, tkey_to_images.values(), tkey_to_targets.values())
    }


@attrs.define
class Task:
    target_classes: list[int]
    training_set: data_utils.Dataset
    test_set: data_utils.Dataset


def make_task(
    training_set: data_utils.Dataset,
    test_set: data_utils.Dataset,
    tkey_to_target_classes: dict[str, list[int]],
):
    key_to_training_set = split_dataset(training_set, tkey_to_target_classes)
    key_to_test_set = split_dataset(test_set, tkey_to_target_classes)

    return {
        tkey: Task(
            target_classes=target_classes,
            training_set=key_to_training_set[tkey],
            test_set=key_to_test_set[tkey],
        )
        for tkey, target_classes in tkey_to_target_classes.items()
    }
