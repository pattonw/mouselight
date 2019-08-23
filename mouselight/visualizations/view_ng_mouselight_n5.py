from funlib.show.neuroglancer import add_layer, ScalePyramid
import argparse
import daisy
import glob
import neuroglancer
import numpy as np
import os
import webbrowser
from swc_parser import _parse_swc
from pathlib import Path
import itertools
import random
import logging

ngid = itertools.count(start=1)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file", "-f", type=str, action="append", help="The path to the container to show"
)
parser.add_argument(
    "--datasets",
    "-d",
    type=str,
    nargs="+",
    action="append",
    help="The datasets in the container to show",
)
parser.add_argument(
    "--synapses",
    "-s",
    type=str,
    action="append",
    help="A numpy npz containing synapse annotations as stored by "
    "synful.gunpowder.ExtractSynapses",
)
parser.add_argument(
    "--time",
    "-t",
    type=int,
    action="store",
    dest="minutes",
    default=0,
    help="How long you want neuroglancer to stay available",
)
parser.add_argument(
    "--output",
    "-o",
    type=str,
    action="store",
    dest="log",
    default="",
    help="Where to output url to",
)

args = parser.parse_args()

print("passed in arguments: {}".format(args))
minutes = args.minutes
print("showing neuroglancer for {} minutes".format(minutes))

if args.log != "":
    logging.basicConfig(level=logging.INFO, filename=args.log)
else:
    logging.basicConfig(level=logging.INFO)

neuroglancer.set_server_bind_address("0.0.0.0")
viewer = neuroglancer.Viewer()

swc_path = Path(
    "/nrs/funke/mouselight-v2/2017-07-02",
    "consensus-neurons-with-machine-centerpoints-labelled-as-swcs/G-002.swc",
)
swc_path = Path(
    "/groups/mousebrainmicro/mousebrainmicro/cluster/2018-07-02/carver/augmented-with-skeleton-nodes-as-swcs/G-002.swc"
)
n5_path = Path(
    "/nrs/funke/mouselight-v2/2018-07-02",
    "consensus-neurons-with-machine-centerpoints-labelled-as-swcs-carved.n5/",
)
transform = Path("/nrs/mouselight/SAMPLES/2018-07-02/transform.txt")


def load_transform(transform_path: Path):
    text = transform_path.open("r").read()
    lines = text.split("\n")
    constants = {}
    for line in lines:
        if len(line) > 0:
            variable, value = line.split(":")
            constants[variable] = float(value)
    spacing = (
        np.array([constants["sx"], constants["sy"], constants["sz"]])
        / 2 ** (constants["nl"] - 1)
        / 1000
    )
    origin = spacing * (
        (np.array([constants["ox"], constants["oy"], constants["oz"]]) // spacing)
        / 1000
    )
    return origin, spacing


def swc_to_voxel_coords(swc_coord, origin, spacing):
    return np.round((swc_coord - origin) / spacing).astype(int)


# swc
neuron_graph = _parse_swc(swc_path)
origin, spacing = load_transform(transform)
voxel_size = spacing

voxel_size_rounded = np.array((10, 3, 3)[::-1])


nodes = []
edges = []
print(len(neuron_graph.nodes))
for node_a, node_b in neuron_graph.edges:
    a = swc_to_voxel_coords(neuron_graph.nodes[node_a]["location"], origin, spacing)
    b = swc_to_voxel_coords(neuron_graph.nodes[node_b]["location"], origin, spacing)

    pos_u = a
    pos_v = b

    nodes.append(
        neuroglancer.EllipsoidAnnotation(
            center=pos_u, radii=(3, 3, 3) / voxel_size, id=next(ngid)
        )
    )
    edges.append(
        neuroglancer.LineAnnotation(point_a=pos_u, point_b=pos_v, id=next(ngid))
    )
    if len(nodes) > 10000:
        break
nodes.append(
    neuroglancer.EllipsoidAnnotation(
        center=pos_v, radii=(1, 1, 1) / voxel_size, id=next(ngid)
    )
)


a = daisy.open_ds(str(n5_path.absolute()), "volume")

with viewer.txn() as s:
    add_layer(s, a, "volume", shader="rgb", c=[0, 0, 0])

with viewer.txn() as s:
    s.layers["edges"] = neuroglancer.AnnotationLayer(
        filter_by_segmentation=False, annotation_color="#add8e6", annotations=edges
    )
    s.layers["nodes"] = neuroglancer.AnnotationLayer(
        filter_by_segmentation=False, annotation_color="#ff00ff", annotations=nodes
    )

url = str(viewer)
logging.info(url)

import time

time.sleep(60 * minutes)

try:
    if minutes < 1:
        input("Press ENTER to exit:")
except:
    pass
