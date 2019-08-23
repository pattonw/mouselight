import daisy
import neuroglancer
import numpy as np
import sys
import itertools

neuroglancer.set_server_bind_address('0.0.0.0')

f = sys.argv[1]
raw = daisy.open_ds(f, 'volumes/raw')
gt = daisy.open_ds(f, 'volumes/gt')
gt_fg = daisy.open_ds(f, 'volumes/gt_fg')
embedding = daisy.open_ds(f, 'volumes/embedding')
fg = daisy.open_ds(f, 'volumes/fg')
maxima = daisy.open_ds(f, 'volumes/maxima')
gradient_embedding = daisy.open_ds(f, 'volumes/gradient_embedding')
gradient_fg = daisy.open_ds(f, 'volumes/gradient_fg')

emst = daisy.open_ds(f, 'emst')
edges_u = daisy.open_ds(f, 'edges_u')
edges_v = daisy.open_ds(f, 'edges_v')

def add(s, a, name, shader=None, visible=True):

    if shader == 'rgb':
        shader="""void main() { emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    kwargs = {}

    if shader is not None:
        kwargs['shader'] = shader

    data = np.expand_dims(a.to_ndarray(), axis=0)
    if len(data.shape) == 4:
        data = np.transpose(data, axes=[1, 0, 2, 3])

    s.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=data,
                offset=a.roi.get_offset()[::-1] + (0,),
                voxel_size=a.voxel_size[::-1] + (1,)
            ),
            visible=visible,
            **kwargs)

embedding.materialize()
mi = np.amin(embedding.data)
ma = np.amax(embedding.data)
embedding.data = (embedding.data - mi)/(ma - mi)
print("Scaled embedding with %.3f"%(ma - mi))

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add(s, raw, 'raw', visible=False)
    add(s, gt, 'gt', visible=False)
    add(s, gt_fg, 'gt_fg', visible=False)
    add(s, embedding, 'embedding', shader='rgb')
    add(s, fg, 'fg', visible=False)
    add(s, maxima, 'maxima')
    add(s, gradient_embedding, 'd_embedding', shader='rgb', visible=False)
    add(s, gradient_fg, 'd_fg', shader='rgb', visible=False)

    mst = []
    node_id = itertools.count(start=1)
    for edge, u, v in zip(emst.to_ndarray(), edges_u.to_ndarray(), edges_v.to_ndarray()):
        print(edge[2])
        if edge[2] > 1.0:
            continue
        pos_u = daisy.Coordinate(u[-3:]*100) + ((0,) + gt.roi.get_offset())
        pos_v = daisy.Coordinate(v[-3:]*100) + ((0,) + gt.roi.get_offset())
        mst.append(neuroglancer.LineAnnotation(
            point_a=pos_u[::-1],
            point_b=pos_v[::-1],
            id=next(node_id)))

    s.layers.append(
        name='mst',
        layer=neuroglancer.AnnotationLayer(annotations=mst)
    )
print(viewer)
