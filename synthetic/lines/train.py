from mahotas import cwatershed
from mala.losses import ultrametric_loss_op
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
import gunpowder as gp
import json
import numpy as np
import skelerator
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)

with open("tensor_names.json", "r") as f:
    tensor_names = json.load(f)


class Synthetic2DSource(gp.BatchProvider):
    def __init__(self, raw, gt, n_objects=2):

        self.raw = raw
        self.gt = gt
        self.n_objects = n_objects

    def setup(self):

        self.provides(
            self.raw,
            gp.ArraySpec(
                roi=gp.Roi((0, 0), (1000, 1000)),
                dtype=np.float32,
                interpolatable=True,
                voxel_size=(1, 1),
            ),
        )
        self.provides(
            self.gt,
            gp.ArraySpec(
                roi=gp.Roi((0, 0), (1000, 1000)),
                dtype=np.uint64,
                interpolatable=False,
                voxel_size=(1, 1),
            ),
        )

    def provide(self, request):

        voxel_size = self.spec[self.raw].voxel_size
        shape = gp.Coordinate((1,) + request[self.raw].roi.get_shape())

        gt_data = np.zeros(shape, dtype=int)
        for i in range(self.n_objects):
            points = np.stack(
                [np.random.randint(0, shape[dim], 2) for dim in range(3)], axis=1
            )
            tree = skelerator.Tree(points)
            skeleton = skelerator.Skeleton(
                tree, [1, 1, 1], "linear", generate_graph=False
            )
            gt_data = skeleton.draw(gt_data, np.array([0, 0, 0]), i + 1)

        gt_data = gt_data[0].astype(np.uint64)
        gt_data = maximum_filter(gt_data, size=2)

        labels = np.unique(gt_data)

        raw_data = (gt_data > 0).astype(np.float32)
        raw_data = np.clip(
            raw_data + np.random.normal(scale=0.1, size=raw_data.shape), 0, 1
        ).astype(np.float32)

        spec = request[self.raw].copy()
        spec.voxel_size = (1, 1)
        raw = gp.Array(raw_data, spec)

        spec = request[self.gt].copy()
        spec.voxel_size = (1, 1)
        gt_crop = (
            request[self.gt].roi - request[self.raw].roi.get_begin()
        ) / voxel_size
        gt_crop = gt_crop.to_slices()
        gt = gp.Array(gt_data[gt_crop], spec)

        batch = gp.Batch()
        batch[self.raw] = raw
        batch[self.gt] = gt

        return batch


emst_name = "PyFuncStateless:0"
edges_u_name = "Gather:0"
edges_v_name = "Gather_1:0"


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

    assert emst.name == emst_name
    assert edges_u.name == edges_u_name
    assert edges_v.name == edges_v_name

    fg_loss = tf.losses.mean_squared_error(gt_fg, fg)

    loss = um_loss + fg_loss

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-5, beta1=0.95, beta2=0.999, epsilon=1e-8
    )

    optimizer = opt.minimize(loss)

    return (loss, optimizer)


def train(n_iterations):

    raw = gp.ArrayKey("RAW")
    gt = gp.ArrayKey("GT")
    gt_fg = gp.ArrayKey("GT_FP")
    embedding = gp.ArrayKey("EMBEDDING")
    fg = gp.ArrayKey("FG")
    maxima = gp.ArrayKey("MAXIMA")
    gradient_embedding = gp.ArrayKey("GRADIENT_EMBEDDING")
    gradient_fg = gp.ArrayKey("GRADIENT_FG")
    emst = gp.ArrayKey("EMST")
    edges_u = gp.ArrayKey("EDGES_U")
    edges_v = gp.ArrayKey("EDGES_V")

    request = gp.BatchRequest()
    request.add(raw, (200, 200))
    request.add(gt, (160, 160))

    snapshot_request = gp.BatchRequest()
    snapshot_request[embedding] = request[gt]
    snapshot_request[fg] = request[gt]
    snapshot_request[gt_fg] = request[gt]
    snapshot_request[maxima] = request[gt]
    snapshot_request[gradient_embedding] = request[gt]
    snapshot_request[gradient_fg] = request[gt]
    snapshot_request[emst] = gp.ArraySpec()
    snapshot_request[edges_u] = gp.ArraySpec()
    snapshot_request[edges_v] = gp.ArraySpec()

    pipeline = (
        Synthetic2DSource(raw, gt)
        + gp.Normalize(raw)
        + gp.tensorflow.Train(
            "train_net",
            optimizer=add_loss,
            loss=None,
            inputs={tensor_names["raw"]: raw, tensor_names["gt_labels"]: gt},
            outputs={
                tensor_names["embedding"]: embedding,
                tensor_names["fg"]: fg,
                "maxima:0": maxima,
                "gt_fg:0": gt_fg,
                emst_name: emst,
                edges_u_name: edges_u,
                edges_v_name: edges_v,
            },
            gradients={
                tensor_names["embedding"]: gradient_embedding,
                tensor_names["fg"]: gradient_fg,
            },
        )
        + gp.Snapshot(
            output_filename="{iteration}.hdf",
            dataset_names={
                raw: "volumes/raw",
                gt: "volumes/gt",
                embedding: "volumes/embedding",
                fg: "volumes/fg",
                maxima: "volumes/maxima",
                gt_fg: "volumes/gt_fg",
                gradient_embedding: "volumes/gradient_embedding",
                gradient_fg: "volumes/gradient_fg",
                emst: "emst",
                edges_u: "edges_u",
                edges_v: "edges_v",
            },
            dataset_dtypes={maxima: np.float32, gt_fg: np.float32},
            every=100,
            additional_request=snapshot_request,
        )
    )

    with gp.build(pipeline):
        for i in range(n_iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":
    train(1000000)
