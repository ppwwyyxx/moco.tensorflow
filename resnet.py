# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorpack.models import (
    BatchNorm, Conv2D, FullyConnected, GlobalAvgPooling, LinearWrap, MaxPooling)
from tensorpack.tfutils import argscope

from data import tf_preprocess


class ResNetModel:
    def __init__(self, num_output=None):
        """
        num_output: int or list[int]: dimension(s) of FC layers in the end
        """
        self.data_format = "NCHW"
        if num_output is not None:
            if not isinstance(num_output, (list, tuple)):
                num_output = [num_output]
        self.num_output = num_output

    def forward(self, image):
        # accept [0-255] BGR NHWC images (from dataflow)
        image = tf_preprocess(image)
        if self.data_format == "NCHW":
            image = tf.transpose(image, [0, 3, 1, 2])
        return self.get_logits(image)

    def get_logits(self, image):
        num_blocks = [3, 4, 6, 3]

        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format), \
                argscope(Conv2D, use_bias=False,
                         kernel_initializer=tf.variance_scaling_initializer(
                             scale=2.0, mode='fan_out', distribution='untruncated_normal')), \
                argscope(BatchNorm, epsilon=1.001e-5):
            logits = (LinearWrap(image)
                      .tf.pad([[0, 0], [0, 0], [3, 3], [3, 3]])
                      .Conv2D('conv0', 64, 7, strides=2, padding='VALID')
                      .apply(self.norm_func, 'conv0')
                      .tf.nn.relu()
                      .tf.pad([[0, 0], [0, 0], [1, 1], [1, 1]])
                      .MaxPooling('pool0', shape=3, stride=2, padding='VALID')
                      .apply(self.resnet_group, 'group0', 64, num_blocks[0], 1)
                      .apply(self.resnet_group, 'group1', 128, num_blocks[1], 2)
                      .apply(self.resnet_group, 'group2', 256, num_blocks[2], 2)
                      .apply(self.resnet_group, 'group3', 512, num_blocks[3], 2)
                      .GlobalAvgPooling('gap')())
            if self.num_output is not None:
                for idx, no in enumerate(self.num_output):
                    logits = FullyConnected(
                        'linear{}_{}'.format(idx, no),
                        logits, no,
                        kernel_initializer=tf.random_normal_initializer(stddev=0.01))
                    if idx != len(self.num_output) - 1:
                        logits = tf.nn.relu(logits)
            return logits

    def norm_func(self, x, name, gamma_initializer=tf.constant_initializer(1.)):
        return BatchNorm(name + '_bn', x, gamma_initializer=gamma_initializer)

    def resnet_group(self, l, name, features, count, stride):
        with tf.variable_scope(name):
            for i in range(0, count):
                with tf.variable_scope('block{}'.format(i)):
                    l = self.bottleneck_block(l, features, stride if i == 0 else 1)
        return l

    def bottleneck_block(self, l, ch_out, stride):
        shortcut = l
        l = Conv2D('conv1', l, ch_out, 1, strides=1)
        l = self.norm_func(l, 'conv1')
        l = tf.nn.relu(l)

        if stride == 1:
            l = Conv2D('conv2', l, ch_out, 3, strides=1)
        else:
            l = tf.pad(l, [[0, 0], [0, 0], [1, 1], [1, 1]])
            l = Conv2D('conv2', l, ch_out, 3, strides=stride, padding='VALID')
        l = self.norm_func(l, 'conv2')
        l = tf.nn.relu(l)

        l = Conv2D('conv3', l, ch_out * 4, 1)
        l = self.norm_func(l, 'conv3')  # pt does not use 0init
        return tf.nn.relu(
            l + self.shortcut(shortcut, ch_out * 4, stride), 'output')

    def shortcut(self, l, n_out, stride):
        n_in = l.get_shape().as_list()[1]
        if n_in != n_out:   # change dimension when channel is not the same
            l = Conv2D('convshortcut', l, n_out, 1, strides=stride)
            l = self.norm_func(l, 'convshortcut')
        return l
