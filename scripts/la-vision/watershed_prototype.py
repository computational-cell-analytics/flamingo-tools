import os

import imageio.v3 as imageio
import napari
import numpy as np
import pandas as pd

from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.segmentation import watershed


def simple_watershed(im, det, radius=8):
    """Use a simple watershed to create speheres.
    """

    # Compute the distance to the detctions.
    seeds = np.zeros(im.shape, dtype="uint8")
    det_idx = tuple(det[ax].values for ax in ["axis-0", "axis-1", "axis-2"])
    seeds[det_idx] = 1
    distances = distance_transform_edt(seeds == 0, sampling=(3.0, 1.887779, 1.887779))
    seeds = label(seeds)

    mask = distances < radius
    return watershed(distances, seeds, mask=mask), distances, seeds


def complex_watershed(im, det, pred, radius=8):
    """More complex waterhsed in combination with network predictions.

    WIP: this does not work well yet.
    """
    fg_pred = pred[0]
    # bd_pred = pred[2]

    _, seeds, distances = simple_watershed(im, det, radius=radius)

    # Ensure everything within five 8 micron of a center is foreground
    fg = np.logical_or(fg_pred > 0.5, distances > radius)

    # TODO find a good hmap!
    hmap = distances

    # Watershed.
    seg = watershed(hmap, markers=seeds, mask=fg, compactness=5)
    return seg, distances, seeds


def main():
    root = "la-vision-sgn-new/detections-v1"
    im = imageio.imread(os.path.join(root, "LaVision-M04_crop_2580-2266-0533_PV.tif"))
    det = pd.read_csv(os.path.join(root, "LaVision-M04_crop_2580-2266-0533_PV.csv"))
    # pred = imageio.imread(os.path.join(root, "LaVision-M04_crop_2580-2266-0533_PRED.tif"))

    seg, distances, seeds = simple_watershed(im, det, radius=12)
    # This does not yet work well.
    # seg, distances, seeds = complex_watershed(im, det, pred)

    v = napari.Viewer()
    v.add_image(im)
    v.add_image(distances, visible=False)
    v.add_labels(seeds, visible=False)
    # v.add_image(pred, visible=False)
    v.add_points(det, visible=False)
    v.add_labels(seg)
    napari.run()


if __name__ == "__main__":
    main()
