from pathlib import Path

import torchvision.transforms.v2 as tf

from . import MetaDataset, download_dataset, make_dataset

TRAIN_URL = 'https://github.com/nghiandxv/datasets/releases/download/cifar-0.0.1/cifar100_train.zip'
TEST_URL = 'https://github.com/nghiandxv/datasets/releases/download/cifar-0.0.1/cifar100_test.zip'
TRAIN_MD5_HASH = '91d63e787989f9512d373cdd6e0f80e1'
TEST_MD5_HASH = '866cf2198497fde9ee870feab8656492'


def init(root: str, force_torch: bool = False):
    if force_torch:
        import torch
        from torchvision.datasets import CIFAR100

        conversion = tf.Compose([tf.PILToTensor(), tf.ToDtype(torch.float32, scale=True)])
        training_set = CIFAR100(root=root, train=True, download=True, transform=conversion)
        test_set = CIFAR100(root=root, train=False, download=True, transform=conversion)
    else:
        root_path = Path(root)
        training_set = make_dataset(download_dataset(TRAIN_URL, root_path, TRAIN_MD5_HASH))
        test_set = make_dataset(download_dataset(TEST_URL, root_path, TEST_MD5_HASH))

    return MetaDataset(
        name='cifar100',
        training_set=training_set,
        test_set=test_set,
        scale_fn=tf.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        augment_fn=tf.Compose([tf.RandomCrop(32, padding=4), tf.RandomHorizontalFlip()]),
    )
