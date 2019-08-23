from pathlib import Path

from neurolight.gunpowder.mouselight_swc_file_source import MouselightSwcFileSource
from neurolight.gunpowder.grow_labels import GrowLabels
from neurolight.gunpowder.rasterize_skeleton import RasterizeSkeleton
from neurolight.gunpowder.get_neuron_pair import GetNeuronPair
from neurolight.gunpowder.fusion_augment import FusionAugment
import gunpowder as gp
from gunpowder import BatchRequest, build, Coordinate
import math
import copy
import time

import logging

import numpy as np

import sys

logging.basicConfig(level=logging.INFO)


SEPERATE_DISTANCE = [100, 130]
SOFT_MASK_SMOOTHING_SIGMA = 30
SCALE_ADD_VOLUME = True


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

        binarized = gp.Array(data=(batch[self.gt].data > 0).astype(np.int32), spec=spec)

        batch[self.gt_binary] = binarized


class Crop(gp.BatchFilter):
    def __init__(self, input_array: gp.ArrayKey, output_array: gp.ArrayKey):

        self.input_array = input_array
        self.output_array = output_array

    def setup(self):

        spec = self.spec[self.input_array].copy()
        self.provides(self.output_array, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):
        input_data = batch[self.input_array].data
        input_spec = batch[self.input_array].spec
        input_roi = input_spec.roi
        output_roi_shape = request[self.output_array].roi.get_shape()
        shift = (input_roi.get_shape() - output_roi_shape) / 2
        output_roi = gp.Roi(shift, output_roi_shape)
        output_data = input_data[
            tuple(
                map(
                    slice,
                    output_roi.get_begin() / input_spec.voxel_size,
                    output_roi.get_end() / input_spec.voxel_size,
                )
            )
        ]
        output_spec = copy.deepcopy(input_spec)
        output_spec.roi = output_roi

        output_array = gp.Array(output_data, output_spec)

        batch[self.output_array] = output_array


input_size = Coordinate([74, 260, 260])
output_size = Coordinate([42, 168, 168])
path_to_data = Path("/nrs/funke/mouselight-v2")

# array keys for data sources
raw = gp.ArrayKey("RAW")
swcs = gp.PointsKey("SWCS")
labels = gp.ArrayKey("LABELS")

# array keys for base volume
raw_base = gp.ArrayKey("RAW_BASE")
labels_base = gp.ArrayKey("LABELS_BASE")
swc_base = gp.PointsKey("SWC_BASE")

# array keys for add volume
raw_add = gp.ArrayKey("RAW_ADD")
labels_add = gp.ArrayKey("LABELS_ADD")
swc_add = gp.PointsKey("SWC_ADD")

# array keys for fused volume
raw_fused = gp.ArrayKey("RAW_FUSED")
labels_fused = gp.ArrayKey("LABELS_FUSED")
swc_fused = gp.PointsKey("SWC_FUSED")
raw_fused_b = gp.ArrayKey("RAW_FUSED_B")
labels_fused_b = gp.ArrayKey("LABELS_FUSED_B")
swc_fused_b = gp.PointsKey("SWC_FUSED_B")

# debugging
masked_base = gp.ArrayKey("MASKED_BASE")
masked_add = gp.ArrayKey("MASKED_ADD")
softmask = gp.ArrayKey("SOFTMASK")
mask_maxed = gp.ArrayKey("MASK_MAXED")
masked_base_b = gp.ArrayKey("MASKED_BASE_B")
masked_add_b = gp.ArrayKey("MASKED_ADD_B")
softmask_b = gp.ArrayKey("SOFTMASK_B")
mask_maxed_b = gp.ArrayKey("MASK_MAXED_B")

# output data
labels_fg = gp.ArrayKey("LABELS_FG")
labels_fg_bin = gp.ArrayKey("LABELS_FG_BIN")

loss_weights = gp.ArrayKey("LOSS_WEIGHTS")

voxel_size = gp.Coordinate((10, 3, 3))
input_size = input_size * voxel_size
output_size = output_size * voxel_size

# add request
request = gp.BatchRequest()
request.add(raw_fused, input_size)
request.add(labels_fused, input_size)
request.add(labels_fg, output_size)
request.add(labels_fg_bin, output_size)
request.add(loss_weights, output_size)

# add snapshot request
# request.add(fg, output_size)
# request.add(labels_fg, output_size)
# request.add(gradient_fg, output_size)
request.add(raw_base, input_size)
request.add(raw_add, input_size)
request.add(labels_base, input_size)
request.add(labels_add, input_size)

# debugging
request.add(masked_base, input_size)
request.add(masked_add, input_size)
request.add(softmask, input_size)
request.add(mask_maxed, input_size)
request.add(masked_base_b, input_size)
request.add(masked_add_b, input_size)
request.add(softmask_b, input_size)
request.add(mask_maxed_b, input_size)

data_sources = tuple(
    (
        gp.N5Source(
            filename=str(
                (
                    filename
                    / "consensus-neurons-with-machine-centerpoints-labelled-as-swcs-carved.n5"
                ).absolute()
            ),
            datasets={raw: "volume"},
            array_specs={
                raw: gp.ArraySpec(
                    interpolatable=True, voxel_size=voxel_size, dtype=np.uint16
                )
            },
        ),
        MouselightSwcFileSource(
            filename=str((filename / "high-res-swcs/G-002.swc").absolute()),
            points=(swcs,),
            scale=voxel_size,
            transpose=(2, 1, 0),
            transform_file=str((filename / "transform.txt").absolute()),
            ignore_human_nodes=False,
        ),
    )
    + gp.MergeProvider()
    + gp.RandomLocation(ensure_nonempty=swcs, ensure_centered=True)
    + RasterizeSkeleton(
        points=swcs,
        array=labels,
        array_spec=gp.ArraySpec(
            interpolatable=False, voxel_size=voxel_size, dtype=np.uint32
        ),
    )
    + GrowLabels(labels, radius=20)
    # augment
    + gp.ElasticAugment([40, 10, 10], [0.25, 1, 1], [0, math.pi / 2.0], subsample=4)
    + gp.SimpleAugment(mirror_only=[1, 2], transpose_only=[1, 2])
    + gp.Normalize(raw)
    + gp.IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001)
    for filename in path_to_data.iterdir()
    if "2018-07-02" in filename.name
)

pipeline = (
    data_sources
    + gp.RandomProvider()
    + GetNeuronPair(
        swcs,
        raw,
        labels,
        (swc_base, swc_add),
        (raw_base, raw_add),
        (labels_base, labels_add),
        seperate_by=SEPERATE_DISTANCE,
        shift_attempts=50,
        request_attempts=10,
    )
    + FusionAugment(
        raw_base,
        raw_add,
        labels_base,
        labels_add,
        swc_base,
        swc_add,
        raw_fused,
        labels_fused,
        swc_fused,
        blend_mode="labels_mask",
        blend_smoothness=SOFT_MASK_SMOOTHING_SIGMA,
        num_blended_objects=0,
        scale_add_volume=True,
        masked_base=masked_base,
        masked_add=masked_add,
        soft_mask=softmask,
        mask_maxed=mask_maxed,
    )
    + FusionAugment(
        raw_base,
        raw_add,
        labels_base,
        labels_add,
        swc_base,
        swc_add,
        raw_fused_b,
        labels_fused_b,
        swc_fused_b,
        blend_mode="labels_mask",
        blend_smoothness=SOFT_MASK_SMOOTHING_SIGMA,
        num_blended_objects=0,
        scale_add_volume=False,
        masked_base=masked_base_b,
        masked_add=masked_add_b,
        soft_mask=softmask_b,
        mask_maxed=mask_maxed_b,
    )
    + gp.PrintProfilingStats(every=1)
    + gp.Snapshot(
        output_filename="snapshot_debug_scale_comparison_{}_{}.hdf".format(
            int(np.mean(SEPERATE_DISTANCE)), "{iteration}"
        ),
        dataset_names={
            raw_fused: "volumes/raw_fused",
            raw_base: "volumes/raw_base",
            raw_add: "volumes/raw_add",
            labels_fused: "volumes/labels_fused",
            labels_base: "volumes/labels_base",
            labels_add: "volumes/labels_add",
            raw_fused_b: "volumes/raw_fused_b",
            labels_fused_b: "volumes/labels_fused_b",
            masked_base: "volumes/masked_base",
            masked_base_b: "volumes/masked_base_b",
            masked_add: "volumes/masked_add",
            masked_add_b: "volumes/masked_add_b",
            softmask: "volumes/softmask",
            softmask_b: "volumes/softmask_b",
            mask_maxed: "volumes/mask_maxed",
            mask_maxed_b: "volumes/mask_maxed_b",
        },
        every=1,
    )
)

with build(pipeline):
    for i in range(1):
        request = BatchRequest(random_seed=i)

        # add request
        request = gp.BatchRequest()
        request.add(raw_fused, input_size)
        request.add(labels_fused, input_size)
        request.add(raw_fused_b, input_size)
        request.add(labels_fused_b, input_size)

        # add snapshot request
        # request.add(fg, output_size)
        # request.add(labels_fg, output_size)
        # request.add(gradient_fg, output_size)
        request.add(raw_base, input_size)
        request.add(raw_add, input_size)
        request.add(labels_base, input_size)
        request.add(labels_add, input_size)

        # debugging
        request.add(masked_base, input_size)
        request.add(masked_add, input_size)
        request.add(softmask, input_size)
        request.add(mask_maxed, input_size)
        request.add(masked_base_b, input_size)
        request.add(masked_add_b, input_size)
        request.add(softmask_b, input_size)
        request.add(mask_maxed_b, input_size)

        pipeline.request_batch(request)
