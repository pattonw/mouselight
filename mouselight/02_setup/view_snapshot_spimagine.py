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
    coordinate = coordinate*2 - 1
    # to xyz
    return coordinate[::-1]

def inspect(raw, roi):

    print("Reading raw data...")

    raw_data = raw.to_ndarray(roi=roi, fill_value=0)

    return spimagine.volshow(
        raw_data,
        stackUnits=raw.voxel_size[1:][::-1])

if __name__ == "__main__":

    filename = sys.argv[1]


    ch1 = daisy.open_ds(
        filename,
        'volumes/ch1')

    a_ch1 = daisy.open_ds(
        filename,
        'volumes/a_ch1')

    b_ch1 = daisy.open_ds(
        filename,
        'volumes/b_ch1')

    soft_mask = daisy.open_ds(
        filename,
        'volumes/soft_mask')

    fused = daisy.Array(
        data=np.array([x.to_ndarray() for x in [ch1, a_ch1, b_ch1, soft_mask]]),
        roi=daisy.Roi(
            (0,) + ch1.roi.get_begin(),
            (4,) + ch1.roi.get_shape()),
        voxel_size=(1,) + ch1.voxel_size)

    inspect(fused, fused.roi)

    input()
