#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import tensorflow as tf

from tensorpack.callbacks import (
    ClassificationError, DataParallelInferenceRunner, EstimatedTimeLeft, InferenceRunner,
    ModelSaver, ScheduledHyperParamSetter, ThroughputTracker)
from tensorpack.dataflow import FakeData
from tensorpack.input_source import QueueInput, StagingInput
from tensorpack.models import BatchNorm, FullyConnected
from tensorpack.tfutils import SaverRestore, argscope, varreplace
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.train import (
    ModelDesc, SyncMultiGPUTrainerReplicated, TrainConfig, launch_train_with_config)
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu

from data import get_imagenet_dataflow
from resnet import ResNetModel


class LinearModel(ModelDesc):
    def __init__(self):
        self.net = ResNetModel(num_output=None)
        self.image_shape = 224

    def inputs(self):
        return [tf.TensorSpec([None, self.image_shape, self.image_shape, 3], tf.uint8, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def compute_loss_and_error(self, logits, label):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        return loss

    def build_graph(self, input, label):
        with argscope(BatchNorm, training=False), \
                varreplace.freeze_variables(skip_collection=True):
            from tensorflow.python.compiler.xla import xla
            feature = xla.compile(lambda: self.net.forward(input))[0]
            # feature = self.net.forward(input)
            feature = tf.stop_gradient(feature)  # double safe
        logits = FullyConnected(
            'linear_cls', feature, 1000,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer())

        tf.nn.softmax(logits, name='prob')
        loss = self.compute_loss_and_error(logits, label)

        # weight decay is 0
        add_moving_summary(loss)
        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.0, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=False)
        return opt


def get_config(model):
    nr_tower = max(get_num_gpu(), 1)
    batch = args.batch // nr_tower

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))

    callbacks = [ThroughputTracker(args.batch)]
    if args.fake:
        data = QueueInput(FakeData(
            [[batch, 224, 224, 3], [batch]], 1000, random=False, dtype='uint8'))
    else:
        data = QueueInput(
            get_imagenet_dataflow(args.data, 'train', batch),
            # use a larger queue
            queue=tf.FIFOQueue(300, [tf.uint8, tf.int32], [[batch, 224, 224, 3], [batch]])
        )
        data = StagingInput(data, nr_stage=1)

        BASE_LR = 30
        SCALED_LR = BASE_LR * (args.batch / 256.0)
        callbacks.extend([
            ModelSaver(),
            EstimatedTimeLeft(),
            ScheduledHyperParamSetter(
                'learning_rate', [
                    (0, min(BASE_LR, SCALED_LR)),
                    (60, SCALED_LR * 1e-1),
                    (70, SCALED_LR * 1e-2),
                    (80, SCALED_LR * 1e-3),
                    (90, SCALED_LR * 1e-4),
                ]),
        ])

        dataset_val = get_imagenet_dataflow(args.data, 'val', 64)
        infs = [ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]
        if nr_tower == 1:
            callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
        else:
            callbacks.append(DataParallelInferenceRunner(
                dataset_val, infs, list(range(nr_tower))))

    if args.load.endswith(".npz"):
        # a released model in npz format
        init = SmartInit(args.load)
    else:
        # a pre-trained checkpoint
        init = SaverRestore(args.load, ignore=("learning_rate", "global_step"))
    return TrainConfig(
        model=model,
        data=data,
        callbacks=callbacks,
        steps_per_epoch=100 if args.fake else 1281167 // args.batch,
        session_init=init,
        max_epoch=100,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='imagenet data dir')
    parser.add_argument('--load', required=True, help='path to pre-trained model')
    parser.add_argument('--fake', help='use FakeData to debug or benchmark this model', action='store_true')
    parser.add_argument('--batch', default=256, type=int, help='total batch size')
    parser.add_argument('--logdir')
    args = parser.parse_args()

    model = LinearModel()

    if args.fake:
        logger.set_logger_dir(os.path.join('train_log', 'tmp'), 'd')
    else:
        if args.logdir is None:
            args.logdir = './moco_lincls'
        logger.set_logger_dir(args.logdir, 'd')

    config = get_config(model)
    trainer = SyncMultiGPUTrainerReplicated(get_num_gpu())
    launch_train_with_config(config, trainer)
