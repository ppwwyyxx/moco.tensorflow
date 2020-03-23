#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import socket
import multiprocessing as mp
import cv2

from tensorpack.dataflow import FakeData, MapData, TestDataSpeed, send_dataflow_zmq
from tensorpack.utils import logger

from data import get_moco_dataflow, get_moco_v1_augmentor, get_moco_v2_augmentor
from zmq_ops import dump_arrays


cv2.setNumThreads(0)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--fake', action='store_true')
    parser.add_argument('--batch', help='per-GPU batch size',
                        default=32, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--v2', action='store_true')
    parser.add_argument('--no-zmq-ops', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if args.fake:
        ds = FakeData(
            [[args.batch, 224, 224, 3], [args.batch, 224, 224, 3]],
            9999999, random=False, dtype=['uint8', 'uint8'])
    else:
        aug = get_moco_v2_augmentor() if args.v2 else get_moco_v1_augmentor()
        ds = get_moco_dataflow(args.data, args.batch, aug)

    logger.info("Serving data on {}".format(socket.gethostname()))

    if args.benchmark:
        ds = MapData(ds, dump_arrays)
        TestDataSpeed(ds, size=99999, warmup=300).start()
    else:
        format = None if args.no_zmq_ops else 'zmq_ops'
        send_dataflow_zmq(
            ds, 'ipc://@imagenet-train-b{}'.format(args.batch),
            hwm=200, format=format, bind=True)
