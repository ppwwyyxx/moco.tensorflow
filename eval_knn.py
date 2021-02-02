#-*- coding: utf-8 -*-

import argparse
import os
import cv2
import tqdm
from collections import Counter

from tensorpack import tfv1 as tf
from tensorpack.utils.stats import Accuracy
from tensorpack.utils import logger
from tensorpack.tfutils import TowerContext, get_default_sess_config
from tensorpack.tfutils.sessinit import SmartInit
from tensorpack.tfutils.varmanip import get_checkpoint_path, get_all_checkpoints
from tensorpack.dataflow import (
    imgaug, DataFromList, BatchData, MultiProcessMapDataZMQ, dataset)

import horovod.tensorflow as hvd
from resnet import ResNetModel
from data import get_basic_augmentor, get_imagenet_dataflow


def build_dataflow(files):
    train_ds = DataFromList(files)
    aug = imgaug.AugmentorList(get_basic_augmentor(isTrain=False))

    def mapper(dp):
        idx, fname, label = dp
        img = cv2.imread(fname)
        img = aug.augment(img)
        return img, idx

    train_ds = MultiProcessMapDataZMQ(train_ds, num_proc=8, map_func=mapper, strict=True)
    train_ds = BatchData(train_ds, local_batch_size)
    train_ds.reset_state()
    return train_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='imagenet data dir')
    parser.add_argument('--batch', default=512, type=int, help='total batch size')
    parser.add_argument('--load', required=True, help='file or directory to evaluate')
    parser.add_argument('--top-k', type=int, default=200, help='top-k in KNN')
    parser.add_argument('--v2', action='store_true', help='use mocov2')
    args = parser.parse_args()

    hvd.init()
    local_batch_size = args.batch // hvd.size()

    train_files = dataset.ILSVRC12Files(args.data, 'train', shuffle=True)
    train_files.reset_state()
    all_train_files = list(train_files)
    all_train_files = all_train_files[:len(all_train_files) // args.batch * args.batch]  # truncate
    num_train_images = len(all_train_files)
    logger.info(f"Creating graph for KNN of {num_train_images} training images ...")
    local_train_files = [(idx, fname, label) for idx, (fname, label) in
                         enumerate(all_train_files) if idx % hvd.size() == hvd.rank()]

    image_input = tf.placeholder(tf.uint8, [None, 224, 224, 3], "image")
    idx_input = tf.placeholder(tf.int64, [None], "image_idx")

    feat_buffer = tf.get_variable("feature_buffer", shape=[num_train_images, 128], trainable=False)
    net = ResNetModel(num_output=(2048, 128) if args.v2 else (128,))
    with TowerContext("", is_training=False):
        feat = net.forward(image_input)
        feat = tf.math.l2_normalize(feat, axis=1)  # Nx128
    all_feat = hvd.allgather(feat)  # GN x 128
    all_idx_input = hvd.allgather(idx_input)  # GN
    update_buffer = tf.scatter_update(feat_buffer, all_idx_input, all_feat)

    dist = tf.matmul(feat, tf.transpose(feat_buffer))  # N x #DS
    _, topk_indices = tf.math.top_k(dist, k=args.top_k)  # Nxtopk

    train_ds = build_dataflow(local_train_files)

    config = get_default_sess_config()
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    def evaluate(checkpoint_file):
        result_file = get_checkpoint_path(checkpoint_file) + f".knn{args.top_k}.txt"
        if os.path.isfile(result_file):
            logger.info(f"Skipping evaluation of {result_file}.")
            return
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            SmartInit(checkpoint_file).init(sess)
            for batch_img, batch_idx in tqdm.tqdm(train_ds, total=len(train_ds)):
                sess.run(update_buffer,
                         feed_dict={image_input: batch_img, idx_input: batch_idx})

            if hvd.rank() == 0:
                acc = Accuracy()
                val_df = get_imagenet_dataflow(args.data, "val", local_batch_size)
                val_df.reset_state()

                for batch_img, batch_label in val_df:
                    topk_indices_pred = sess.run(topk_indices, feed_dict={image_input: batch_img})
                    for indices, gt in zip(topk_indices_pred, batch_label):
                        pred = [all_train_files[k][1] for k in indices]
                        top_pred = Counter(pred).most_common(1)[0]
                        acc.feed(top_pred[0] == gt, total=1)
                logger.info(f"Accuracy of {checkpoint_file}: {acc.accuracy} out of {acc.total}")
                with open(result_file, "w") as f:
                    f.write(str(acc.accuracy))

    if os.path.isdir(args.load):
        for fname, _ in get_all_checkpoints(args.load):
            logger.info(f"Evaluating {fname} ...")
            evaluate(fname)
    else:
        evaluate(args.load)
