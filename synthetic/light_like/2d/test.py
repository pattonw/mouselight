from mala.losses import ultrametric_loss_op
import neurolight as nl
import gunpowder as gp
import json
import numpy as np
import tensorflow as tf
import logging

from typing import List
import math

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# logging.basicConfig(level=logging.DEBUG, filename="log.txt")
logging.basicConfig(level=logging.INFO)

INPUT_SHAPE = [204, 204]
OUTPUT_SHAPE = [120, 120]

LABEL_RADII = [2, 3]
RAW_RADII = [1, 2]
RAW_INTENSITIES = [0.7, 0.9]
NOISE_VAR = 0.05

SKEL_GEN_RADIUS = 10
THETAS = np.array([0.15, 0.3]) * math.pi
SPLIT_PS = [0.98, 0.02]
N_OBJS = 2


class PrintDataTypes(gp.BatchFilter):
    def process(self, batch, request):
        for key, array in batch.arrays.items():
            print("array %s has dtype %s" % (key, array.data.dtype))


class LabelToFloat32(gp.BatchFilter):
    def __init__(self, array: gp.ArrayKey, intensities: List[float] = [1.0]):
        self.array = array
        self.intensities = intensities

    def setup(self):
        self.enable_autoskip()
        spec = self.spec[self.array].copy()
        spec.dtype = np.float32
        self.updates(self.array, spec)

    def process(self, batch, request):
        array = batch[self.array]
        intensity_data = np.zeros_like(array.data, dtype=np.float32)
        for i, label in enumerate(np.unique(array.data)):
            if label == 0:
                continue
            mask = array.data == label
            intensity_data[mask] = np.maximum(
                intensity_data[mask], self.intensities[i % len(self.intensities)]
            )
        spec = array.spec
        spec.dtype = np.float32
        batch[self.array] = gp.Array(intensity_data, spec)


with open("tensor_names.json", "r") as f:
    tensor_names = json.load(f)


emst_name = "PyFuncStateless:0"
edges_u_name = "GatherV2:0"
edges_v_name = "GatherV2_1:0"


def add_loss(graph):

    # k, h, w
    embedding = graph.get_tensor_by_name(tensor_names["embedding"])

    # h, w
    fg = graph.get_tensor_by_name(tensor_names["fg"])

    # h, w
    gt_labels = graph.get_tensor_by_name(tensor_names["gt_labels"])

    # h, w
    gt_fg = tf.greater(gt_labels, 0, name="gt_fg")

    # h, w
    shape = tuple(fg.get_shape().as_list())

    # 1, 1, h, w
    maxima = tf.nn.pool(
        tf.reshape(fg, (1, 1) + shape),
        [10, 10],
        "MAX",
        "SAME",
        strides=[1, 1],
        data_format="NCHW",
    )
    # h, w
    maxima = tf.reshape(tf.equal(fg, maxima), shape, name="maxima")

    # 1, k, h, w
    embedding = tf.reshape(embedding, (1,) + tuple(embedding.get_shape().as_list()))
    # k, 1, h, w
    embedding = tf.transpose(embedding, perm=[1, 0, 2, 3])

    um_loss, emst, edges_u, edges_v, _ = ultrametric_loss_op(
        embedding, gt_labels, mask=maxima, coordinate_scale=0.01
    )

    print("um_loss: {}".format(um_loss))

    assert emst.name == emst_name, "{} != {}".format(emst.name, emst_name)
    assert edges_u.name == edges_u_name, "{} != {}".format(edges_u.name, edges_u_name)
    assert edges_v.name == edges_v_name, "{} != {}".format(edges_v.name, edges_v_name)

    fg_loss = tf.losses.mean_squared_error(gt_fg, fg)

    loss = um_loss + fg_loss

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-5, beta1=0.95, beta2=0.999, epsilon=1e-8
    )

    optimizer = opt.minimize(loss)

    return (loss, optimizer)


def train(n_iterations):

    point_trees = gp.PointsKey("POINT_TREES")
    labels = gp.ArrayKey("LABELS")
    raw = gp.ArrayKey("RAW")
    # gt_fg = gp.ArrayKey("GT_FG")
    # embedding = gp.ArrayKey("EMBEDDING")
    # fg = gp.ArrayKey("FG")
    # maxima = gp.ArrayKey("MAXIMA")
    # gradient_embedding = gp.ArrayKey("GRADIENT_EMBEDDING")
    # gradient_fg = gp.ArrayKey("GRADIENT_FG")
    # emst = gp.ArrayKey("EMST")
    # edges_u = gp.ArrayKey("EDGES_U")
    # edges_v = gp.ArrayKey("EDGES_V")

    request = gp.BatchRequest()
    request.add(raw, INPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    request.add(labels, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    request.add(point_trees, INPUT_SHAPE)

    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw, INPUT_SHAPE)
    # snapshot_request.add(embedding, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    # snapshot_request.add(fg, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    # snapshot_request.add(gt_fg, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    # snapshot_request.add(maxima, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    # snapshot_request.add(
    #     gradient_embedding, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1))
    # )
    # snapshot_request.add(gradient_fg, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    # snapshot_request[emst] = gp.ArraySpec()
    # snapshot_request[edges_u] = gp.ArraySpec()
    # snapshot_request[edges_v] = gp.ArraySpec()

    pipeline = (
        nl.SyntheticLightLike(
            point_trees,
            dims=2,
            r=SKEL_GEN_RADIUS,
            n_obj=N_OBJS,
            thetas=THETAS,
            split_ps=SPLIT_PS,
        )
        # + gp.SimpleAugment()
        # + gp.ElasticAugment([10, 10], [0.1, 0.1], [0, 2.0 * math.pi], spatial_dims=2)
        + nl.RasterizeSkeleton(
            point_trees,
            labels,
            gp.ArraySpec(
                roi=gp.Roi((None,) * 2, (None,) * 2),
                voxel_size=gp.Coordinate((1, 1)),
                dtype=np.uint64,
            ),
        )
        + gp.Copy(labels, raw)
        + nl.GrowLabels(labels, radii=LABEL_RADII)
        + nl.GrowLabels(raw, radii=RAW_RADII)
        + LabelToFloat32(raw, intensities=RAW_INTENSITIES)
        + gp.NoiseAugment(raw, var=NOISE_VAR)
        # + gp.PreCache(cache_size=40, num_workers=10)
        # + gp.tensorflow.Train(
        #     "train_net",
        #     optimizer=add_loss,
        #     loss=None,
        #     inputs={tensor_names["raw"]: raw, tensor_names["gt_labels"]: labels},
        #     outputs={
        #         tensor_names["embedding"]: embedding,
        #         tensor_names["fg"]: fg,
        #         "maxima:0": maxima,
        #         "gt_fg:0": gt_fg,
        #         emst_name: emst,
        #         edges_u_name: edges_u,
        #         edges_v_name: edges_v,
        #     },
        #     gradients={
        #         tensor_names["embedding"]: gradient_embedding,
        #         tensor_names["fg"]: gradient_fg,
        #     },
        # )
        + gp.Snapshot(
            output_filename="{iteration}.hdf",
            dataset_names={
                raw: "volumes/raw",
                labels: "volumes/labels",
                point_trees: "point_trees",
                # embedding: "volumes/embedding",
                # fg: "volumes/fg",
                # maxima: "volumes/maxima",
                # gt_fg: "volumes/gt_fg",
                # gradient_embedding: "volumes/gradient_embedding",
                # gradient_fg: "volumes/gradient_fg",
                # emst: "emst",
                # edges_u: "edges_u",
                # edges_v: "edges_v",
            },
            # dataset_dtypes={maxima: np.float32, gt_fg: np.float32},
            every=100,
            additional_request=snapshot_request,
        )
        + gp.PrintProfilingStats(every=10)
    )

    with gp.build(pipeline):
        for i in range(n_iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":
    train(10)
