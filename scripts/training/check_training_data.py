import os
from glob import glob

import imageio.v3 as imageio
import napari
import numpy as np

root = "/home/pape/Work/data/moser/lightsheet"


def check_visually(check_downsampled=False):
    if check_downsampled:
        images = sorted(glob(os.path.join(root, "images_s2", "*.tif")))
        masks = sorted(glob(os.path.join(root, "masks_s2", "*.tif")))
    else:
        images = sorted(glob(os.path.join(root, "images", "*.tif")))
        masks = sorted(glob(os.path.join(root, "masks", "*.tif")))
    assert len(images) == len(masks)

    for im, mask in zip(images, masks):
        print(im)

        vol = imageio.imread(im)
        seg = imageio.imread(mask).astype("uint32")

        v = napari.Viewer()
        v.add_image(vol)
        v.add_labels(seg)
        napari.run()


def check_labels():
    masks = sorted(glob(os.path.join(root, "masks", "*.tif")))
    for mask_path in masks:
        labels = imageio.imread(mask_path)
        n_labels = len(np.unique(labels))
        print(mask_path, n_labels)


if __name__ == "__main__":
    check_visually(True)
    # check_labels()
