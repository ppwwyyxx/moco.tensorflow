# -*- coding: utf-8 -*-

import numpy as np
import cv2
import multiprocessing as mp
import tensorflow as tf

from tensorpack.dataflow import (
    BatchData, MultiProcessMapAndBatchDataZMQ, MultiProcessRunnerZMQ, MultiThreadMapData, dataset,
    imgaug)


cv2.setNumThreads(0)


def get_moco_v1_augmentor():
    augmentors = [
        imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.2, 1.)),
        imgaug.RandomApplyAug(imgaug.Grayscale(rgb=False, keepshape=True), 0.2),
        imgaug.ToFloat32(),
        imgaug.RandomOrderAug(
            [imgaug.BrightnessScale((0.6, 1.4)),
             imgaug.Contrast((0.6, 1.4), rgb=False),
             imgaug.Saturation(0.4, rgb=False),
             # 72 = 180*0.4
             imgaug.Hue(range=(-72, 72), rgb=False)
             ]),
        imgaug.ToUint8(),
        imgaug.Flip(horiz=True),
    ]
    return augmentors


def get_moco_v2_augmentor():
    augmentors = [
        imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.2, 1.)),
        imgaug.ToFloat32(),
        imgaug.RandomApplyAug(
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4)),
                 imgaug.Contrast((0.6, 1.4), rgb=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # 18 = 180*0.1
                 imgaug.Hue(range=(-18, 18), rgb=False)
                 ]), 0.8),
        imgaug.RandomApplyAug(imgaug.Grayscale(rgb=False, keepshape=True), 0.2),
        imgaug.RandomApplyAug(
            # 11 = 0.1*224//2
            imgaug.GaussianBlur(size_range=(11, 12), sigma_range=[0.1, 2.0]), 0.5),
        imgaug.ToUint8(),
        imgaug.Flip(horiz=True),
    ]
    return augmentors


class MoCoMapper:
    def __init__(self, augs):
        self.augs = augs

    def __call__(self, dp):
        fname, _ = dp  # throw away the label
        img = cv2.imread(fname)
        img1 = self.augs.augment(img)
        img2 = self.augs.augment(img)
        return [img1, img2]


def get_moco_dataflow(datadir, batch_size, augmentors):
    """
    Dataflow for training MOCO.
    """
    augmentors = imgaug.AugmentorList(augmentors)
    parallel = min(30, mp.cpu_count())  # tuned on a 40-CPU 80-core machine
    ds = dataset.ILSVRC12Files(datadir, 'train', shuffle=True)
    ds = MultiProcessMapAndBatchDataZMQ(ds, parallel, MoCoMapper(augmentors), batch_size, buffer_size=5000)
    return ds


def get_basic_augmentor(isTrain):
    interpolation = cv2.INTER_LINEAR
    if isTrain:
        augmentors = [
            imgaug.GoogleNetRandomCropAndResize(),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, interp=interpolation),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors


def get_imagenet_dataflow(datadir, name, batch_size, parallel=None):
    """
    Get a standard imagenet training/evaluation dataflow, for linear classifier tuning.
    """
    assert name in ['train', 'val']
    isTrain = name == 'train'
    assert datadir is not None
    augmentors = get_basic_augmentor(isTrain)
    augmentors = imgaug.AugmentorList(augmentors)
    if parallel is None:
        parallel = min(50, mp.cpu_count())

    def mapper(dp):
        fname, label = dp
        img = cv2.imread(fname)
        img = augmentors.augment(img)
        return img, label

    if isTrain:
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=True)
        ds = MultiProcessMapAndBatchDataZMQ(ds, parallel, mapper, batch_size, buffer_size=7000)
    else:
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=False)
        ds = MultiThreadMapData(ds, parallel, mapper, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = MultiProcessRunnerZMQ(ds, 1)
    return ds


def tf_preprocess(image):  # normalize BGR images
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        mean = [0.485, 0.456, 0.406]    # rgb
        std = [0.229, 0.224, 0.225]
        mean = mean[::-1]
        std = std[::-1]
        image_mean = tf.constant(mean, dtype=tf.float32) * 255.
        image_std = tf.constant(std, dtype=tf.float32) * 255.
        image = (image - image_mean) / image_std
        return image


if __name__ == '__main__':
    from tensorpack.dataflow import TestDataSpeed
    import sys
    df = get_imagenet_dataflow(sys.argv[1], 'train', 32)

    TestDataSpeed(df, size=99999999, warmup=300).start()
