from __future__ import print_function
import numpy as np
import gunpowder as gp
from neurolight.gunpowder import FusionAugment, RasterizeSkeleton
from neurolight.gunpowder.mouselight_swc_file_source import MouselightSwcFileSource
from neurolight.gunpowder.get_neuron_pair import GetNeuronPair
from neurolight.gunpowder.recenter import Recenter
from neurolight.gunpowder.grow_labels import GrowLabels
import json
import logging
import tensorflow as tf
import sys
import math
from pathlib import Path
import logging
from spimagine import volshow
import networkx as nx


import neuroglancer
import numpy as np
import itertools
import networkx as nx


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

    voxel_size = gp.Coordinate((10, 3, 3))
    input_size = gp.Coordinate(net_config["input_shape"]) * voxel_size * 2

    # add request
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(swcs, input_size)

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
                        / "consensus-neurons-with-machine-centerpoints-labelled-as-swcs/G-002.swc"
                    ).absolute()
                ),
                points=(swcs,),
                scale=voxel_size,
                transpose=(2, 1, 0),
                transform_file=str((filename / "transform.txt").absolute()),
            ),
        )
        + gp.MergeProvider()
        + gp.RandomLocation(ensure_nonempty=swcs, ensure_centered=True)
        for filename in Path(sample_dir).iterdir()
        if "2018-08-01" in filename.name
    )

    pipeline = data_sources + gp.RandomProvider()

    with gp.build(pipeline):

        print("Starting training...")
        for i in range(max_iteration - trained_until):
            batch = pipeline.request_batch(request)
            vis_points_with_array(
                batch[raw].data, points_to_graph(batch[swcs].data), np.array(voxel_size)
            )


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

    logging.basicConfig(level=logging.DEBUG)

    iteration = int(sys.argv[1])
    train_until(iteration)
