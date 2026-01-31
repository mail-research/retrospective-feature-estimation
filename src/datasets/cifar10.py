from pathlib import Path

import torchvision.transforms.v2 as tf

from . import MetaDataset, download_dataset, make_dataset

TRAIN_URL = 'https://github.com/nghiandxv/datasets/releases/download/cifar-0.0.1/cifar10_train.zip'
TEST_URL = 'https://github.com/nghiandxv/datasets/releases/download/cifar-0.0.1/cifar10_test.zip'
TRAIN_MD5_HASH = 'a37e8b26c7955882ae6a716e8e3feb9a'
TEST_MD5_HASH = '17574914976f0ba4c7f443f344426a4a'


def init(root: str, force_torch: bool = False):
    if force_torch:
        import torch
        from torchvision.datasets import CIFAR10

        conversion = tf.Compose([tf.PILToTensor(), tf.ToDtype(torch.float32, scale=True)])
        training_set = CIFAR10(root=root, train=True, download=True, transform=conversion)
        test_set = CIFAR10(root=root, train=False, download=True, transform=conversion)
    else:
        root_path = Path(root)
        training_set = make_dataset(download_dataset(TRAIN_URL, root_path, TRAIN_MD5_HASH))
        test_set = make_dataset(download_dataset(TEST_URL, root_path, TEST_MD5_HASH))

    return MetaDataset(
        name='cifar10',
        training_set=training_set,
        test_set=test_set,
        scale_fn=tf.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        augment_fn=tf.Compose([tf.RandomCrop(32, padding=4), tf.RandomHorizontalFlip()]),
    )
