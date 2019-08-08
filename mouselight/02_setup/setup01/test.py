from __future__ import print_function

import numpy as np
import json
import logging
import tensorflow as tf
import sys
import math
from pathlib import Path
import networkx as nx
import copy

import gunpowder as gp
import numpy as np
import itertools

logging.basicConfig(
    level=logging.DEBUG,
    filename="log.txt",
)

from neurolight.gunpowder import FusionAugment, RasterizeSkeleton
from neurolight.gunpowder.mouselight_swc_file_source import MouselightSwcFileSource
from neurolight.gunpowder.get_neuron_pair import GetNeuronPair
from neurolight.gunpowder.recenter import Recenter
from neurolight.gunpowder.grow_labels import GrowLabels


"""
def vis_points_with_array(raw: np.ndarray, points: nx.DiGraph, voxel_size: np.ndarray):
    print([point for point in points.nodes.values()])
    if len(raw.shape) == 4:
        raw = raw[0, :, :, :]

    ngid = itertools.count(start=1)

    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()

    nodes = []
    edges = []
    pos_v = None

    for node_a, node_b in points.edges:
        a = points.nodes[node_a]["location"]
        b = points.nodes[node_b]["location"]

        pos_u = a / voxel_size
        pos_v = b / voxel_size

        nodes.append(
            neuroglancer.EllipsoidAnnotation(
                center=pos_u, radii=(3, 3, 3) / voxel_size, id=next(ngid)
            )
        )
        edges.append(
            neuroglancer.LineAnnotation(point_a=pos_u, point_b=pos_v, id=next(ngid))
        )
    if pos_v is not None:
        nodes.append(
            neuroglancer.EllipsoidAnnotation(
                center=pos_v, radii=(1, 1, 1) / voxel_size, id=next(ngid)
            )
        )

    max_raw = np.max(raw)
    min_raw = np.min(raw)
    print("values in range: {}-{}".format(min_raw, max_raw))
    diff_raw = max_raw - min_raw

    raw = ((raw - min_raw) / float(diff_raw) * 255).astype("uint8")

    with viewer.txn() as s:
        s.layers["raw"] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=raw.transpose([2, 1, 0]), voxel_size=voxel_size
            )
        )
        s.layers["edges"] = neuroglancer.AnnotationLayer(
            voxel_size=voxel_size,
            filter_by_segmentation=False,
            annotation_color="#add8e6",
            annotations=edges,
        )
        s.layers["nodes"] = neuroglancer.AnnotationLayer(
            voxel_size=voxel_size,
            filter_by_segmentation=False,
            annotation_color="#ff00ff",
            annotations=nodes,
        )
        position = np.array(raw.shape) // 2
        s.navigation.position.voxelCoordinates = tuple(position)
    print(viewer)

    input("done?")
"""

sample_dir = "/nrs/funke/mouselight-v2"


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
        print(input_roi, output_roi)
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


with open("train_net_config.json", "r") as f:
    net_config = json.load(f)
with open("train_net_names.json", "r") as f:
    net_names = json.load(f)


def train_until(max_iteration):

    # get the latest checkpoint
    if tf.train.latest_checkpoint("."):
        trained_until = int(tf.train.latest_checkpoint(".").split("_")[-1])
    else:
        trained_until = 0
        if trained_until >= max_iteration:
            return

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

    # output data
    fg = gp.ArrayKey("FG")
    labels_fg = gp.ArrayKey("LABELS_FG")
    labels_fg_bin = gp.ArrayKey("LABELS_FG_BIN")

    gradient_fg = gp.ArrayKey("GRADIENT_FG")
    loss_weights = gp.ArrayKey("LOSS_WEIGHTS")

    voxel_size = gp.Coordinate((10, 3, 3))
    input_size = gp.Coordinate(net_config["input_shape"]) * voxel_size
    output_size = gp.Coordinate(net_config["output_shape"]) * voxel_size

    # add request
    request = gp.BatchRequest()
    request.add(raw_fused, input_size)
    request.add(labels_fused, input_size)
    request.add(swc_fused, input_size)
    request.add(labels_fg, output_size)
    request.add(labels_fg_bin, output_size)
    request.add(loss_weights, output_size)

    # add snapshot request
    # request.add(fg, output_size)
    # request.add(labels_fg, output_size)
    request.add(gradient_fg, output_size)
    request.add(raw_base, input_size)
    request.add(raw_add, input_size)
    request.add(labels_base, input_size)
    request.add(labels_add, input_size)
    request.add(swc_base, input_size)
    request.add(swc_add, input_size)

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
                filename=str(
                    (
                        filename
                        / "consensus-neurons-with-machine-centerpoints-labelled-as-swcs"
                    ).absolute()
                ),
                points=(swcs,),
                scale=voxel_size,
                transpose=(2, 1, 0),
                transform_file=str((filename / "transform.txt").absolute()),
                ignore_human_nodes=True
            ),
        )
        + gp.MergeProvider()
        + gp.RandomLocation(
            ensure_nonempty=swcs, ensure_centered=True, voxel_size=voxel_size
        )
        + RasterizeSkeleton(
            points=swcs,
            array=labels,
            array_spec=gp.ArraySpec(
                interpolatable=False, voxel_size=voxel_size, dtype=np.uint32
            ),
        )
        + GrowLabels(labels, radius=10)
        # augment
        + gp.ElasticAugment(
            [40, 10, 10],
            [0.25, 1, 1],
            [0, math.pi / 2.0],
            subsample=4,
            voxel_size=voxel_size,
        )
        + gp.SimpleAugment(mirror_only=[1, 2], transpose_only=[1, 2])
        + gp.Normalize(raw)
        + gp.IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001)
        for filename in Path(sample_dir).iterdir()
        if "2018-08-01" in filename.name  # or "2018-07-02" in filename.name
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
            seperate_by=150,
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
            blend_smoothness=10,
            num_blended_objects=0,
        )
        + Crop(labels_fused, labels_fg)
        + BinarizeGt(labels_fg, labels_fg_bin)
        + gp.BalanceLabels(labels_fg_bin, loss_weights)
        # train
        + gp.PreCache(cache_size=40, num_workers=10)
        + gp.tensorflow.Train(
            "./train_net",
            optimizer=net_names["optimizer"],
            loss=net_names["loss"],
            inputs={
                net_names["raw"]: raw_fused,
                net_names["labels_fg"]: labels_fg_bin,
                net_names["loss_weights"]: loss_weights,
            },
            outputs={net_names["fg"]: fg},
            gradients={net_names["fg"]: gradient_fg},
            save_every=100000,
        )
        + gp.Snapshot(
            output_filename="snapshot_{iteration}.hdf",
            dataset_names={
                raw_fused: "volumes/raw_fused",
                raw_base: "volumes/raw_base",
                raw_add: "volumes/raw_add",
                labels_fused: "volumes/labels_fused",
                labels_base: "volumes/labels_base",
                labels_add: "volumes/labels_add",
                labels_fg_bin: "volumes/labels_fg_bin",
                fg: "volumes/pred_fg",
                gradient_fg: "volumes/gradient_fg",
            },
            every=100,
        )
        + gp.PrintProfilingStats(every=10)
    )

    with gp.build(pipeline):

        logging.info("Starting training...")
        for i in range(max_iteration - trained_until):
            logging.info("requesting batch {}".format(i))
            batch = pipeline.request_batch(request)
            """
            vis_points_with_array(
                batch[raw_fused].data,
                points_to_graph(batch[swc_fused].data),
                np.array(voxel_size),
            )"""


def points_to_graph(points):
    g = nx.DiGraph()
    for point_id, point in points.items():
        g.add_node(point_id, location=point.location)
        if (
            point.parent_id is not None
            and point.parent_id != point_id
            and point.parent_id != -1
            and point.parent_id in points
        ):
            g.add_edge(point_id, point.parent_id)
    return g


if __name__ == "__main__":

    # logging.basicConfig(level=logging.INFO, filename="log.txt")
    logging.info("Starting training!")

    iteration = int(sys.argv[1])
    train_until(iteration)
