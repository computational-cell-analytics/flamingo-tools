import os
from glob import glob

import napari
import pandas as pd
import tifffile

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs"
ANNOTATION_FOLDERS = ["AnnotationsEK", "AnnotationsAMD", "AnnotationLR"]
COLOR = ["green", "yellow", "orange"]


def _match_annotations(image_path):
    prefix = os.path.basename(image_path).split("_")[:3]
    prefix = "_".join(prefix)

    annotations = {}
    for annotation_folder in ANNOTATION_FOLDERS:
        all_annotations = glob(os.path.join(ROOT, annotation_folder, "*.csv"))
        matches = [ann for ann in all_annotations if os.path.basename(ann).startswith(prefix)]
        if len(matches) != 1:
            continue
        annotation_path = matches[0]

        annotation = pd.read_csv(annotation_path)[["axis-0", "axis-1", "axis-2"]].values
        annotations[annotation_folder] = annotation

    return annotations


def compare_annotations(image_path):
    annotations = _match_annotations(image_path)

    image = tifffile.memmap(image_path)
    v = napari.Viewer()
    v.add_image(image)
    for i, (name, annotation) in enumerate(annotations.items()):
        v.add_points(annotation, name=name, face_color=COLOR[i])
    v.title = os.path.basename(image_path)
    napari.run()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+")
    args = parser.parse_args()

    if args.images is None:
        image_paths = sorted(glob(os.path.join(ROOT, "*.tif")))
    else:
        image_paths = args.images

    for image_path in image_paths:
        compare_annotations(image_path)


if __name__ == "__main__":
    main()
