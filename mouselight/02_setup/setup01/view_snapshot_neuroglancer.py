import neuroglancer
import numpy as np
import itertools
import networkx as nx


def vis_points_with_array(raw: np.ndarray, points: nx.DiGraph, voxel_size: np.ndarray):
    ngid = itertools.count(start=1)

    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()

    nodes = []
    edges = []

    for node_a, node_b in points.edges:
        a = points.nodes[node_a]["location"][::-1]
        b = points.nodes[node_b]["location"][::-1]

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
    nodes.append(
        neuroglancer.EllipsoidAnnotation(
            center=pos_v, radii=(1, 1, 1) / voxel_size, id=next(ngid)
        )
    )

    print(raw.shape)

    max_raw = np.max(raw)
    min_raw = np.min(raw)
    diff_raw = max_raw - min_raw

    try:
        raw = ((raw - min_raw) / float(diff_raw) * 255).astype("uint8")
    except Exception as e:
        print(min_raw, max_raw)
        raise e

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

