from pathlib import Path

import torchvision.transforms.v2 as tf

from . import MetaDataset, download_dataset, make_dataset

TRAIN_URL = 'https://github.com/nghiandxv/datasets/releases/download/tiny_imagenet-0.0.1/tiny_imagenet_200_train.zip'
TEST_URL = 'https://github.com/nghiandxv/datasets/releases/download/tiny_imagenet-0.0.1/tiny_imagenet_200_test.zip'
TRAIN_MD5_HASH = '8cc0777061ae0a4a26ec145f73f91806'
TEST_MD5_HASH = 'a7535dab2090cedb0c59b3a0ba07521f'


def init(root: str, force_torch: bool = False):  # force_torch do nothing here
    root_path = Path(root)
    return MetaDataset(
        name='tiny_imagenet',
        training_set=make_dataset(download_dataset(TRAIN_URL, root_path, TRAIN_MD5_HASH)),
        test_set=make_dataset(download_dataset(TEST_URL, root_path, TEST_MD5_HASH)),
        scale_fn=tf.Normalize(mean=[0.4802, 0.4480, 0.3975], std=[0.2764, 0.2689, 0.2816]),
        augment_fn=tf.Compose([tf.RandomCrop(64, padding=4), tf.RandomHorizontalFlip()]),
    )
