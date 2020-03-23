#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import subprocess
import tensorflow as tf
from tensorflow.python.compiler.xla import xla

from tensorpack.callbacks import (
    Callback, EstimatedTimeLeft, ModelSaver, ScheduledHyperParamSetter, ThroughputTracker)
from tensorpack.dataflow import FakeData
from tensorpack.input_source import QueueInput, TFDatasetInput, ZMQInput
from tensorpack.models import BatchNorm, l2_regularizer, regularize_cost
from tensorpack.tfutils import argscope, varreplace
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.train import (
    HorovodTrainer, ModelDesc, TrainConfig, launch_train_with_config)
from tensorpack.utils import logger

import horovod.tensorflow as hvd
from resnet import ResNetModel

BASE_LR = 0.03


def num_gpu():
    return hvd.size()


def allgather(tensor, name):
    tensor = tf.identity(tensor, name=name + "_HVD")
    return hvd.allgather(tensor)


def batch_shuffle(tensor):  # nx...
    total, rank = hvd.size(), hvd.rank()
    batch_size = tf.shape(tensor)[0]
    with tf.device('/cpu:0'):
        all_idx = tf.range(total * batch_size)
        shuffle_idx = tf.random.shuffle(all_idx)
        shuffle_idx = hvd.broadcast(shuffle_idx, 0)
        my_idxs = tf.slice(shuffle_idx, [rank * batch_size], [batch_size])

    all_tensor = allgather(tensor, 'batch_shuffle_key')  # gn x ...
    return tf.gather(all_tensor, my_idxs), shuffle_idx


def batch_unshuffle(key_feat, shuffle_idxs):
    rank = hvd.rank()
    inv_shuffle_idx = tf.argsort(shuffle_idxs)
    batch_size = tf.shape(key_feat)[0]
    my_idxs = tf.slice(inv_shuffle_idx, [rank * batch_size], [batch_size])
    all_key_feat = allgather(key_feat, "batch_unshuffle_feature")  # gn x c
    return tf.gather(all_key_feat, my_idxs)


class MOCOModel(ModelDesc):
    def __init__(self, batch_size, feature_dims=(128,), temp=0.07):
        self.batch_size = batch_size
        self.feature_dim = feature_dims[-1]
        # NOTE: implicit assume queue_size % (batch_size * GPU) ==0
        self.queue_size = 65536
        self.temp = temp

        self.net = ResNetModel(num_output=feature_dims)
        self.image_shape = 224

    def inputs(self):
        return [tf.TensorSpec([self.batch_size, self.image_shape, self.image_shape, 3], tf.uint8, 'query'),
                tf.TensorSpec([self.batch_size, self.image_shape, self.image_shape, 3], tf.uint8, 'key')]

    def build_graph(self, query, key):
        # setup queue
        queue_init = tf.math.l2_normalize(
            tf.random.normal([self.queue_size, self.feature_dim]), axis=1)
        queue = tf.get_variable('queue', initializer=queue_init, trainable=False)
        queue_ptr = tf.get_variable(
            'queue_ptr',
            [], initializer=tf.zeros_initializer(),
            dtype=tf.int64, trainable=False)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, queue)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, queue_ptr)

        # query encoder
        q_feat = self.net.forward(query)  # NxC
        q_feat = tf.math.l2_normalize(q_feat, axis=1)

        # key encoder
        shuffled_key, shuffle_idxs = batch_shuffle(key)
        shuffled_key.set_shape([self.batch_size, None, None, None])
        with tf.variable_scope("momentum_encoder"), \
                varreplace.freeze_variables(skip_collection=True), \
                argscope(BatchNorm, ema_update='skip'):  # don't maintain EMA (will not be used at all)
            key_feat = xla.compile(lambda: self.net.forward(shuffled_key))[0]
            # key_feat = self.net.forward(shuffled_key)
        key_feat = tf.math.l2_normalize(key_feat, axis=1)  # NxC
        key_feat = batch_unshuffle(key_feat, shuffle_idxs)
        key_feat = tf.stop_gradient(key_feat)

        # loss
        l_pos = tf.reshape(tf.einsum('nc,nc->n', q_feat, key_feat), (-1, 1))  # nx1
        l_neg = tf.einsum('nc,kc->nk', q_feat, queue)  # nxK
        logits = tf.concat([l_pos, l_neg], axis=1)  # nx(1+k)
        logits = logits * (1 / self.temp)
        labels = tf.zeros(self.batch_size, dtype=tf.int64)  # n
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.math.argmax(logits, axis=1), labels), tf.float32), name='train-acc')

        # update queue (depend on l_neg)
        with tf.control_dependencies([l_neg]):
            queue_push_op = self.push_queue(queue, queue_ptr, key_feat)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, queue_push_op)

        wd_loss = regularize_cost(".*", l2_regularizer(1e-4), name='l2_regularize_loss')
        add_moving_summary(acc, loss, wd_loss)
        total_cost = tf.add_n([loss, wd_loss], name='cost')
        return total_cost

    def push_queue(self, queue, queue_ptr, item):
        # queue: KxC
        # item: NxC
        item = allgather(item, 'queue_gather')  # GN x C
        batch_size = tf.shape(item, out_type=tf.int64)[0]
        end_queue_ptr = queue_ptr + batch_size

        inds = tf.range(queue_ptr, end_queue_ptr, dtype=tf.int64)
        with tf.control_dependencies([inds]):
            queue_ptr_update = tf.assign(queue_ptr, end_queue_ptr % self.queue_size)
        queue_update = tf.scatter_update(queue, inds, item)
        return tf.group(queue_update, queue_ptr_update)

    def optimizer(self):
        if args.v2:
            # cosine LR in v2
            gs = tf.train.get_or_create_global_step()
            total_steps = 1281167 // args.batch * 200
            lr = BASE_LR * 0.5 * (1 + tf.cos(gs / total_steps * np.pi))
        else:
            lr = tf.get_variable('learning_rate', initializer=0.0, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        return opt


class UpdateMomentumEncoder(Callback):
    _chief_only = False  # execute it in every worker
    momentum = 0.999

    def _setup_graph(self):
        nontrainable_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
        all_vars = {v.name: v for v in tf.global_variables() + tf.local_variables()}

        # find variables of encoder & momentum encoder
        self._var_mapping = {}  # var -> mom var
        momentum_prefix = "momentum_encoder/"
        for mom_var in nontrainable_vars:
            if momentum_prefix in mom_var.name:
                q_encoder_name = mom_var.name.replace(momentum_prefix, "")
                q_encoder_var = all_vars[q_encoder_name]
                assert q_encoder_var not in self._var_mapping
                if not q_encoder_var.trainable:  # don't need to copy EMA
                    continue
                self._var_mapping[q_encoder_var] = mom_var

        logger.info(f"Found {len(self._var_mapping)} pairs of matched variables.")

        assign_ops = [tf.assign(mom_var, var) for var, mom_var in self._var_mapping.items()]
        self.assign_op = tf.group(*assign_ops, name="initialize_momentum_encoder")

        update_ops = [tf.assign_add(mom_var, (var - mom_var) * (1 - self.momentum))
                      for var, mom_var in self._var_mapping.items()]
        self.update_op = tf.group(*update_ops, name="update_momentum_encoder")

    def _before_train(self):
        logger.info("Copying encoder to momentum encoder ...")
        self.assign_op.run()

    def _trigger_step(self):
        self.update_op.run()


def get_config(model):
    input_sig = model.get_input_signature()
    nr_tower = max(num_gpu(), 1)
    batch = args.batch // nr_tower
    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))

    callbacks = [
        ThroughputTracker(args.batch),
        UpdateMomentumEncoder()
    ]

    if args.fake:
        data = QueueInput(FakeData(
            [x.shape for x in input_sig], 1000, random=False, dtype='uint8'))
    else:
        zmq_addr = 'ipc://@imagenet-train-b{}'.format(batch)
        data = ZMQInput(zmq_addr, 25, bind=False)

        dataset = data.to_dataset(input_sig).repeat().prefetch(15)
        dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))
        data = TFDatasetInput(dataset)

        callbacks.extend([
            ModelSaver(),
            EstimatedTimeLeft(),
        ])

        if not args.v2:
            # step-wise LR in v1
            SCALED_LR = BASE_LR * (args.batch / 256.0)
            callbacks.append(
                ScheduledHyperParamSetter(
                    'learning_rate', [
                        (0, min(BASE_LR, SCALED_LR)),
                        (120, SCALED_LR * 1e-1),
                        (160, SCALED_LR * 1e-2)
                    ]))
            if SCALED_LR > BASE_LR:
                callbacks.append(
                    ScheduledHyperParamSetter(
                        'learning_rate', [(0, BASE_LR), (5, SCALED_LR)], interp='linear'))

    return TrainConfig(
        model=model,
        data=data,
        callbacks=callbacks,
        steps_per_epoch=100 if args.fake else 1281167 // args.batch,
        max_epoch=200,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='imagenet data dir')
    parser.add_argument('--fake', help='use FakeData to debug or benchmark this model', action='store_true')
    parser.add_argument('--batch', default=256, type=int, help='total batch size')
    parser.add_argument('--v2', action='store_true', help='train mocov2')
    parser.add_argument('--logdir')
    args = parser.parse_args()

    hvd.init()

    local_batch_size = args.batch // num_gpu()
    if args.v2:
        model = MOCOModel(batch_size=local_batch_size, feature_dims=(2048, 128), temp=0.2)
    else:
        model = MOCOModel(batch_size=local_batch_size, feature_dims=(128,), temp=0.07)

    if hvd.rank() == 0:
        if args.fake:
            logger.set_logger_dir('fake_train_log', 'd')
        else:
            if args.logdir is None:
                args.logdir = 'train_log'
            logger.set_logger_dir(args.logdir, 'd')
    logger.info("Rank={}, Local Rank={}, Size={}".format(hvd.rank(), hvd.local_rank(), hvd.size()))

    if not args.fake and hvd.local_rank() == 0:
        # start data serving process
        script = os.path.realpath(os.path.join(os.path.dirname(__file__), "serve-data.py"))
        v2_flag = "--v2" if args.v2 else ""
        cmd = f"taskset --cpu-list 0-29 {script} --data {args.data} --batch {local_batch_size} {v2_flag}"
        log_prefix = os.path.join(args.logdir, "data." + str(hvd.rank()))
        logger.info("Launching command: " + cmd)
        pid = subprocess.Popen(
            cmd,
            shell=True,
            stdout=open(log_prefix + ".stdout", "w"),
            stderr=open(log_prefix + ".stderr", "w"))

    config = get_config(model)
    trainer = HorovodTrainer(average=True)
    launch_train_with_config(config, trainer)
