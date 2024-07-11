import os
from glob import glob

import imageio.v3 as imageio
from skimage.transform import rescale

DATA_ROOT = "/scratch-grete/usr/nimcpape/data/moser/lightsheet"


def prepare_data_2x():
    image_paths = sorted(glob(os.path.join(DATA_ROOT, "images", "*.tif")))
    label_paths = sorted(glob(os.path.join(DATA_ROOT, "masks", "*.tif")))

    out_images = os.path.join(DATA_ROOT, "images_s2")
    os.makedirs(out_images, exist_ok=True)
    out_labels = os.path.join(DATA_ROOT, "masks_s2")
    os.makedirs(out_labels, exist_ok=True)

    for imp, labelp in zip(image_paths, label_paths):
        im = imageio.imread(imp)
        im = rescale(im, 0.5, preserve_range=True, order=3).astype(im.dtype)
        out_im = os.path.join(out_images, os.path.basename(imp))
        imageio.imwrite(out_im, im, compression="zlib")

        mask = imageio.imread(labelp)
        mask = rescale(mask, 0.5, preserve_range=True, order=0, anti_aliasing=False).astype(mask.dtype)
        out_mask = os.path.join(out_labels, os.path.basename(labelp))
        imageio.imwrite(out_mask, mask, compression="zlib")


prepare_data_2x()
