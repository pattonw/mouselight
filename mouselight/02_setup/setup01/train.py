from __future__ import print_function
import numpy as np
import gunpowder as gp
from neurolight.gunpowder import FusionAugment, SwcFileSource, RasterizeSkeleton, GrowLabels, GetNeuronPair, Recenter
import json
import logging
import tensorflow as tf
import sys
import math

sample_directory = "/nrs/funke/mouselight-v2"

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

        batch[self.gt_binary] = binarized


with open('train_net_config.json', 'r') as f:
    net_config = json.load(f)
with open('train_net_names.json', 'r') as f:
    net_names = json.load(f)


def train_until(max_iteration):

    # get the latest checkpoint
    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
        if trained_until >= max_iteration:
            return

    # array keys for data sources
    raw = gp.ArrayKey('RAW')
    swcs = gp.PointsKey('swcs')
    labels = gp.ArrayKey('LABELS')

    # array keys for base volume
    raw_base = gp.ArrayKey('RAW_BASE')
    labels_base = gp.ArrayKey('LABELS_BASE')
    swc_base = gp.PointsKey('SWC_BASE')

    # array keys for add volume
    raw_add = gp.ArrayKey('RAW_ADD')
    labels_add = gp.ArrayKey('LABELS_ADD')
    swc_add = gp.PointsKey('SWC_ADD')

    # array keys for fused volume
    raw_fused = gp.ArrayKey('RAW_FUSED')
    labels_fused = gp.ArrayKey('LABELS_FUSED')
    swc_fused = gp.PointsKey('SWC_FUSED')

    # output data
    fg = gp.ArrayKey('FG')
    gradient_fg = gp.ArrayKey('GRADIENT_FG')
    loss_weights = gp.ArrayKey('LOSS_WEIGHTS')

    voxel_size = gp.Coordinate((4, 1, 1))
    input_size = gp.Coordinate(net_config['input_shape']) * voxel_size
    output_size = gp.Coordinate(net_config['output_shape']) * voxel_size

    # add request
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_fg, output_size)
    request.add(loss_weights, output_size)

    # add snapshot request
    snapshot_request = gp.BatchRequest()
    snapshot_request.add(fg, output_size)
    snapshot_request.add(labels_fg, output_size)
    snapshot_request.add(gradient_fg, output_size)
    snapshot_request.add(raw_base, input_size)
    snapshot_request.add(raw_add, input_size)
    snapshot_request.add(labels_base, input_size)
    snapshot_request.add(labels_add, input_size)

    # data source for "base" volume
    data_sources = tuple(
        (
            gp.N5Source(
                str((filename/"raw").absolute()),
                datasets={
                    raw: 'volume',
                },
                array_specs={
                    raw: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size, dtype=np.uint16),
                }
            ),
            SwcFileSource(
                filename=str((filename/"swcs").absolute()),
                points=(swcs),
                scale=voxel_size,
                transpose=(2,1,0),
            )
        ) +
        gp.MergeProvider() +
        gp.RandomLocation(ensure_centered=swcs) +
        RasterizeSkeleton(
            points=swcs,
            array=labels,
            array_spec=gp.ArraySpec(interpolatable=False, voxel_size=voxel_size, dtype=np.uint32),
        ) +
        GrowLabels(labels, radius=5)
        for filename in Path(sample_dir).iterdir()
    ) 

    pipeline = (
            data_sources +
            RandomLocation() +
            GetNeuronPair() +
            FusionAugment(
                raw_base,
                raw_add,
                labels_base,
                labels_add,
                raw,
                labels,
                blend_mode='intensity',
                blend_smoothness=10,
                num_blended_objects=0) +

            # augment
            gp.ElasticAugment(
                [40,10,10],
                [0.25,1,1],
                [0,math.pi/2.0],
                subsample=4) +
            gp.SimpleAugment(mirror_only=[1, 2], transpose_only=[1, 2]) +
            gp.Normalize(raw) +
            gp.IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001) +

            BinarizeGt(labels, labels_fg) +
            gp.BalanceLabels(labels_fg, loss_weights) +

            # train
            gp.PreCache(
                cache_size=40,
                num_workers=10) +
            gp.tensorflow.Train(
                './train_net',
                optimizer=net_names['optimizer'],
                loss=net_names['loss'],
                inputs={
                    net_names['raw']: raw,
                    net_names['labels_fg']: labels_fg,
                    net_names['loss_weights']: loss_weights,
                },
                outputs={
                    net_names['fg']: fg,
                },
                gradients={
                    net_names['fg']: gradient_fg,
                },
                save_every=100000) +

            # visualize
            gp.Snapshot(
                output_filename='snapshot_{iteration}.hdf',
                dataset_names={
                    raw: 'volumes/raw',
                    raw_base: 'volumes/raw_base',
                    raw_add: 'volumes/raw_add',
                    labels: 'volumes/labels',
                    labels_base: 'volumes/labels_base',
                    labels_add: 'volumes/labels_add',
                    fg: 'volumes/fg',
                    labels_fg: 'volumes/labels_fg',
                    gradient_fg: 'volumes/gradient_fg',
                },
                additional_request=snapshot_request,
                every=100) +

            gp.PrintProfilingStats(every=100)
    )

    with gp.build(pipeline):

        print("Starting training...")
        for i in range(max_iteration - trained_until):
            pipeline.request_batch(request)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    iteration = int(sys.argv[1])
    train_until(iteration)
