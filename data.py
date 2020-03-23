# -*- coding: utf-8 -*-

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageFilter

from tensorpack.dataflow import (
    BatchData, MultiProcessMapAndBatchDataZMQ, MultiProcessRunnerZMQ, MultiThreadMapData, dataset,
    imgaug)


cv2.setNumThreads(0)


class RandomGrayScale(imgaug.PhotometricAugmentor):
    def __init__(self, prob):
        super().__init__()
        self._init(locals())

    def _get_augment_params(self, _):
        return self._rand_range() < self.prob  # do

    def _augment(self, img, do):
        if do:
            m = cv2.COLOR_BGR2GRAY
            grey = cv2.cvtColor(img, m)
            return np.stack([grey] * 3, axis=2)
        else:
            return img


class RandomGaussionBlurPIL(imgaug.PhotometricAugmentor):
    def __init__(self, sigma):
        super().__init__()
        self._init(locals())

    def _get_augment_params(self, _):
        sigma = self._rand_range(self.sigma[0], self.sigma[1])
        return sigma

    def _augment(self, img, sigma):
        img = Image.fromarray(img.astype("uint8"))
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.array(img)


class TorchvisionCropAndResize(imgaug.ImageAugmentor):
    """
    Unfortunately it's slightly different from the classical
    GoogleNet CropAndResize in fb.resnet.torch.
    """
    def __init__(self, crop_area_fraction=(0.08, 1.),
                 aspect_ratio_range=(0.75, 1.333),
                 target_shape=224, interp=cv2.INTER_LINEAR):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(*self.crop_area_fraction) * area
            log_ratio = (np.log(self.aspect_ratio_range[0]), np.log(self.aspect_ratio_range[1]))
            aspectR = np.exp(self.rng.uniform(*log_ratio))
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if hh <= h and ww <= w:
                x1 = self.rng.randint(0, w - ww + 1)
                y1 = self.rng.randint(0, h - hh + 1)
                return imgaug.TransformList([
                    imgaug.CropTransform(y1, x1, hh, ww),
                    imgaug.ResizeTransform(hh, ww, self.target_shape, self.target_shape, interp=self.interp)
                ])
        in_ratio = float(w) / float(h)
        ratio = self.aspect_ratio_range
        if (in_ratio < min(ratio)):
            ww = w
            hh = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            hh = h
            ww = int(round(h * max(ratio)))
        else:
            ww, hh = w, h
        y1 = (h - hh) // 2
        x1 = (w - ww) // 2
        return imgaug.TransformList([
            imgaug.CropTransform(y1, x1, hh, ww),
            imgaug.ResizeTransform(hh, ww, self.target_shape, self.target_shape, interp=self.interp)
        ])


def get_moco_v1_augmentor():
    augmentors = [
        TorchvisionCropAndResize(crop_area_fraction=(0.2, 1.)),
        RandomGrayScale(0.2),
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
        TorchvisionCropAndResize(crop_area_fraction=(0.2, 1.)),
        imgaug.ToFloat32(),
        imgaug.RandomApplyAug(
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4)),
                 imgaug.Contrast((0.6, 1.4), rgb=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # 18 = 180*0.1
                 imgaug.Hue(range=(-18, 18), rgb=False)
                 ]), 0.8),
        RandomGrayScale(0.2),
        imgaug.RandomApplyAug(
            RandomGaussionBlurPIL([0.1, 2.0]), 0.5),
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
    parallel = 30  # tuned on a 40-CPU 80-core machine
    ds = dataset.ILSVRC12Files(datadir, 'train', shuffle=True)
    ds = MultiProcessMapAndBatchDataZMQ(ds, parallel, MoCoMapper(augmentors), batch_size, buffer_size=5000)
    return ds


def get_basic_augmentor(isTrain):
    interpolation = cv2.INTER_LINEAR
    if isTrain:
        augmentors = [
            TorchvisionCropAndResize(),
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
        parallel = 60

    def mapf(dp):
        fname, label = dp
        img = cv2.imread(fname)
        img = augmentors.augment(img)
        return img, label

    if isTrain:
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=True)
        ds = MultiProcessMapAndBatchDataZMQ(ds, parallel, mapf, batch_size, buffer_size=7000)
    else:
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=False)
        ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=2000, strict=True)
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
