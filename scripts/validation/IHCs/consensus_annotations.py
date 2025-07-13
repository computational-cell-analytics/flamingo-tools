import os
from glob import glob

import pandas as pd
from flamingo_tools.validation import create_consensus_annotations

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationIHCs"
ANNOTATION_FOLDERS = ["Annotations_AMD", "Annotations_EK", "Annotations_LR"]

OUTPUT_FOLDER = os.path.join(ROOT, "consensus_annotation")
COLOR = ["blue", "yellow", "orange"]


def match_annotations(image_path):
    annotations = {}
    prefix = os.path.basename(image_path).split("_")[:3]
    prefix = "_".join(prefix)

    annotations = {}
    for annotation_folder in ANNOTATION_FOLDERS:
        all_annotations = glob(os.path.join(ROOT, annotation_folder, "*.csv"))
        matches = [ann for ann in all_annotations if os.path.basename(ann).startswith(prefix)]
        assert len(matches) == 1
        annotation_path = matches[0]
        annotations[annotation_folder] = annotation_path

    return annotations


def consensus_annotations(image_path, check):
    annotation_paths = match_annotations(image_path)
    assert len(annotation_paths) == len(ANNOTATION_FOLDERS)

    # I tried first with a matching distnce of 8, but that is too conservative.
    # A mathing distance of 16 seems better, but might still need to refine this.
    matching_distance = 16.0
    consensus_annotations, unmatched_annotations = create_consensus_annotations(
        annotation_paths, matching_distance=matching_distance, min_matches_for_consensus=2,
    )
    fname = os.path.basename(image_path)

    if check:
        import napari
        import tifffile

        consensus_annotations = consensus_annotations[["axis-0", "axis-1", "axis-2"]].values
        unmatched_annotators = unmatched_annotations.annotator.values
        unmatched_annotations = unmatched_annotations[["axis-0", "axis-1", "axis-2"]].values

        image = tifffile.imread(image_path)
        v = napari.Viewer()
        v.add_image(image)
        v.add_points(consensus_annotations, face_color="green")
        v.add_points(
            unmatched_annotations,
            properties={"annotator": unmatched_annotators},
            face_color="annotator",
            face_color_cycle=COLOR,  # TODO reorder
        )
        v.title = os.path.basename(fname)
        napari.run()
    else:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        consensus_annotations = consensus_annotations[["axis-0", "axis-1", "axis-2"]]
        consensus_annotations.insert(0, "annotator", ["consensus"] * len(consensus_annotations))
        unmatched_annotations = unmatched_annotations[["axis-0", "axis-1", "axis-2", "annotator"]]
        annotations = pd.concat([consensus_annotations, unmatched_annotations])
        output_path = os.path.join(OUTPUT_FOLDER, fname.replace(".tif", ".csv"))
        annotations.to_csv(output_path, index=False)
        print("Saved to", output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.images is None:
        image_paths = sorted(glob(os.path.join(ROOT, "*.tif")))
    else:
        image_paths = args.images

    for image_path in image_paths:
        consensus_annotations(image_path, args.check)


if __name__ == "__main__":
    main()
