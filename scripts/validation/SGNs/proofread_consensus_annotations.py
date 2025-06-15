import os
from glob import glob

import napari
import pandas as pd
import tifffile

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs"
CONSENSUS_FOLDER = os.path.join(ROOT, "Consensus")
COLOR = ["blue", "yellow", "orange"]


def proofread_consensus_annotations(image_path, annotation_path, color_by_annotator):
    image = tifffile.memmap(image_path)
    annotations = pd.read_csv(annotation_path)

    consensus_annotations = annotations[annotations.annotator == "consensus"][["axis-0", "axis-1", "axis-2"]].values
    unmatched_annotations = annotations[annotations.annotator != "consensus"]

    unmatched_annotators = unmatched_annotations.annotator.values
    unmatched_annotations = unmatched_annotations[["axis-0", "axis-1", "axis-2"]].values

    image = tifffile.imread(image_path)
    v = napari.Viewer()
    v.add_image(image)
    v.add_points(consensus_annotations, face_color="green")
    if color_by_annotator:
        v.add_points(
            unmatched_annotations,
            properties={"annotator": unmatched_annotators},
            face_color="annotator",
            face_color_cycle=COLOR,  # TODO reorder
        )
    else:
        v.add_points(unmatched_annotations)
    fname = os.path.basename(annotation_path)
    v.title = os.path.basename(fname)
    napari.run()


# TODO enable skipping the ones already stored in the output folder and specifying a different root path
# TODO set reasonable contrast limits
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--color_by_annotator", action="store_true")
    args = parser.parse_args()

    annotations = sorted(glob(os.path.join(CONSENSUS_FOLDER, "*.csv")))
    for annotation_path in annotations:
        fname = os.path.basename(annotation_path)
        image_path = os.path.join(ROOT, fname.replace(".csv", ".tif"))
        proofread_consensus_annotations(image_path, annotation_path, args.color_by_annotator)


if __name__ == "__main__":
    main()
