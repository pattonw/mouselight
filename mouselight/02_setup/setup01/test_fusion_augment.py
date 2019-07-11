from __future__ import print_function
import numpy as np
import gunpowder as gp
from neurolight.gunpowder import *
import json
import logging
import tensorflow as tf
import sys
import math
import os
import random

files = [
    '/nrs/funke/mouselight/2018-04-25/swc-carved.h5',
    '/nrs/funke/mouselight/2018-04-25/swc-carved.h5',
    # '/home/maisl/data/mouselight/2018-04-25/swc-carved.h5',
    # '/home/maisl/data/mouselight/2018-04-25/swc-carved.h5',
]


class BinarizeGt(gp.BatchFilter):

    def __init__(self, gt, gt_binary):

        self.gt = gt
        self.gt_binary = gt_binary

    def setup(self):

        spec = self.spec[self.gt].copy()
        spec.dtype = np.uint8
        self.provides(self.gt_binary, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):

        spec = batch[self.gt].spec.copy()
        spec.dtype = np.int32

        binarized = gp.Array(
            data=(batch[self.gt].data > 0).astype(np.int32),
            spec=spec)
        print('sum fg: ', np.sum(binarized.data))

        batch[self.gt_binary] = binarized


with open('train_net_config.json', 'r') as f:
    net_config = json.load(f)
with open('train_net_names.json', 'r') as f:
    net_names = json.load(f)


def train_until(max_iteration, return_intermediates=False):

    # get the latest checkpoint
    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
        if trained_until >= max_iteration:
            return

    # input data
    ch1 = gp.ArrayKey('CH1')
    ch2 = gp.ArrayKey('CH2')
    swc = gp.PointsKey('SWC')
    swc_env = gp.PointsKey('SWC_ENV')
    swc_center = gp.PointsKey('SWC_CENTER')
    gt = gp.ArrayKey('GT')
    gt_fg = gp.ArrayKey('GT_FG')

    # show fusion augment batches
    if return_intermediates:

        a_ch1 = gp.ArrayKey('A_CH1')
        a_ch2 = gp.ArrayKey('A_CH2')
        b_ch1 = gp.ArrayKey('B_CH1')
        b_ch2 = gp.ArrayKey('B_CH2')
        soft_mask = gp.ArrayKey('SOFT_MASK')

    # output data
    fg = gp.ArrayKey('FG')
    gradient_fg = gp.ArrayKey('GRADIENT_FG')
    loss_weights = gp.ArrayKey('LOSS_WEIGHTS')

    voxel_size = gp.Coordinate((4, 1, 1))
    input_size = gp.Coordinate(net_config['input_shape']) * voxel_size
    output_size = gp.Coordinate(net_config['output_shape']) * voxel_size

    # add request
    request = gp.BatchRequest()
    request.add(ch1, input_size)
    request.add(ch2, input_size)
    request.add(swc, input_size)
    request.add(swc_center, output_size)
    request.add(gt, output_size)
    request.add(gt_fg, output_size)
    # request.add(loss_weights, output_size)

    if return_intermediates:

        request.add(a_ch1, input_size)
        request.add(a_ch2, input_size)
        request.add(b_ch1, input_size)
        request.add(b_ch2, input_size)
        request.add(soft_mask, input_size)

    # add snapshot request
    snapshot_request = gp.BatchRequest()
    # snapshot_request[fg] = request[gt]
    # snapshot_request[gt_fg] = request[gt]
    # snapshot_request[gradient_fg] = request[gt]

    data_sources = tuple()
    data_sources += tuple(

        (Hdf5ChannelSource(
            file,
            datasets={
                ch1: '/volume',
                ch2: '/volume',
            },
            channel_ids={
                ch1: 0,
                ch2: 1,
            },
            data_format='channels_last',
            array_specs={
                ch1: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size, dtype=np.uint16),
                ch2: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size, dtype=np.uint16),
            }
        ),
         SwcSource(
             filename=file,
             dataset='/reconstruction',
             points=(swc_center, swc),
             return_env=True,
             scale=voxel_size
         )) +
        gp.MergeProvider() +
        gp.RandomLocation(ensure_nonempty=swc_center) +
        RasterizeSkeleton(points=swc,
                          array=gt,
                          array_spec=gp.ArraySpec(interpolatable=False, voxel_size=voxel_size, dtype=np.uint32),
                          points_env=swc_env,
                          iteration=10)
        for file in files
    )

    snapshot_datasets = {}

    if return_intermediates:

        snapshot_datasets = {
            ch1: 'volumes/ch1',
            ch2: 'volumes/ch2',
            a_ch1: 'volumes/a_ch1',
            a_ch2: 'volumes/a_ch2',
            b_ch1: 'volumes/b_ch1',
            b_ch2: 'volumes/b_ch2',
            soft_mask: 'volumes/soft_mask',
            gt: 'volumes/gt',
            fg: 'volumes/fg',
            gt_fg: 'volumes/gt_fg',
            gradient_fg: 'volumes/gradient_fg',
        }

    else:

        snapshot_datasets = {
            ch1: 'volumes/ch1',
            ch2: 'volumes/ch2',
            gt: 'volumes/gt',
            fg: 'volumes/fg',
            gt_fg: 'volumes/gt_fg',
            gradient_fg: 'volumes/gradient_fg',
        }

    pipeline = (

            data_sources +
            #gp.RandomProvider() +
            FusionAugment(ch1, ch2, gt, smoothness=1, return_intermediate=return_intermediates) +

            # augment
            #gp.ElasticAugment(...) +
            #gp.SimpleAugment() +
            gp.Normalize(ch1) +
            gp.Normalize(ch2) +
            gp.Normalize(a_ch1) +
            gp.Normalize(a_ch2) +
            gp.Normalize(b_ch1) +
            gp.Normalize(b_ch2) +
            gp.IntensityAugment(ch1, 0.9, 1.1, -0.001, 0.001) +
            gp.IntensityAugment(ch2, 0.9, 1.1, -0.001, 0.001) +

            BinarizeGt(gt, gt_fg) +

            # visualize
            gp.Snapshot(
                output_filename='snapshot_{iteration}.hdf',
                dataset_names=snapshot_datasets,
                additional_request=snapshot_request,
                every=20) +

            gp.PrintProfilingStats(every=1000)

    )

    with gp.build(pipeline):

        print("Starting training...")
        for i in range(max_iteration - trained_until):
            pipeline.request_batch(request)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    random.seed(42)

    if len(sys.argv) <= 1:

        iteration = 100
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        return_intermediates = True

    else:
        iteration = int(sys.argv[1])
        return_intermediates = False
        if len(sys.argv) > 2:
            return_intermediates = True if str(sys.argv[2]) == 'return_intermediates' else False

    train_until(1, return_intermediates)
