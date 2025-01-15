import argparse
import os

import imageio.v3 as imageio
import napari
import numpy as np

from train_distance_unet import get_image_and_label_paths
from tqdm import tqdm

# Root folder on my laptop.
# This is just for convenience, so that I don't have to pass
# the root argument during development.
ROOT_CP = "/home/pape/Work/data/moser/lightsheet"


def check_visually(images, labels):
    for im, label in tqdm(zip(images, labels), total=len(images)):

        vol = imageio.imread(im)
        seg = imageio.imread(label).astype("uint32")

        v = napari.Viewer()
        v.add_image(vol, name="pv-channel")
        v.add_labels(seg, name="annotations")
        folder, name = os.path.split(im)
        folder = os.path.basename(folder)
        v.title = f"{folder}/{name}"
        napari.run()


def check_labels(images, labels):
    for label_path in labels:
        labels = imageio.imread(label_path)
        n_labels = len(np.unique(labels))
        print(label_path, n_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", "-i", help="The root folder with the annotated training crops.",
        default=ROOT_CP,
    )
    parser.add_argument("--check_labels", "-l", action="store_true")
    args = parser.parse_args()
    root = args.root

    images, labels = get_image_and_label_paths(root)

    check_visually(images, labels)
    if args.check_labels:
        check_labels(images, labels)


if __name__ == "__main__":
    main()
