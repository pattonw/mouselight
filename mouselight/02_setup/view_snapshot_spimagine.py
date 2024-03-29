import spimagine
import daisy
import sys
import numpy as np


def to_spimagine_coords(coordinate, roi):

    # relative to ROI begin
    coordinate -= roi.get_begin()[1:]
    # relative to ROI size in [0, 1]
    coordinate /= np.array(roi.get_shape()[1:], dtype=np.float32)
    # relative to ROI size in [-1, 1]
    coordinate = coordinate * 2 - 1
    # to xyz
    return coordinate[::-1]


def inspect(raw, roi):

    print("Reading raw data...")

    raw_data = raw.to_ndarray(roi=roi, fill_value=0)

    return spimagine.volshow(raw_data, stackUnits=raw.voxel_size[1:][::-1])


if __name__ == "__main__":

    filename = sys.argv[1]

    raw_base = daisy.open_ds(filename, "volumes/raw_base")
    print(raw_base.roi)

    labels_base = daisy.open_ds(filename, "volumes/labels_base")
    print(labels_base.roi)

    raw_add = daisy.open_ds(filename, "volumes/raw_add")
    print(raw_add.roi)

    labels_add = daisy.open_ds(filename, "volumes/labels_add")
    print(labels_add.roi)

    raw_fused = daisy.open_ds(filename, "volumes/raw_fused")
    print(raw_fused.roi)

    labels_fused = daisy.open_ds(filename, "volumes/labels_fused")
    print(labels_fused.roi)

    all_data = daisy.Array(
        data=np.array(
            [
                x.to_ndarray()[0, :, :, :] if len(x.data.shape) == 4 else x.to_ndarray()
                for x in [
                    raw_base,
                    labels_base,
                    raw_add,
                    labels_add,
                    raw_fused,
                    labels_fused,
                ]
            ]
        ),
        roi=daisy.Roi((0,) + raw_base.roi.get_begin(), (6,) + raw_base.roi.get_shape()),
        voxel_size=(1,) + raw_base.voxel_size,
    )

    inspect(all_data, all_data.roi)

    input()
