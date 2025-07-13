import os
from glob import glob

import pandas as pd
from flamingo_tools.validation import create_consensus_annotations

IMAGE_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs/for_consensus_annotation"  # noqa
ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs/final_annotations"  # noqa

ANNOTATION_FOLDERS = ["AnnotationsAMD", "AnnotationsEK", "AnnotationsLR"]
COLOR = ["blue", "yellow", "orange"]
CONSENSUS_ANNOTATIONS = "consensus_annotations"
OUTPUT_FOLDER = os.path.join(ROOT, "final_consensus_annotations")


def match_annotations(image_path, annotation_folders):
    annotations = {}
    prefix = os.path.basename(image_path).split("_")[:3]
    prefix = "_".join(prefix)

    annotations = {}
    for annotation_folder in annotation_folders:
        all_annotations = glob(os.path.join(ROOT, annotation_folder, "*.csv"))
        matches = [
            ann for ann in all_annotations if (os.path.basename(ann).startswith(prefix) and "negative" not in ann)
        ]
        if len(matches) != 1:
            breakpoint()
        assert len(matches) == 1
        annotation_path = matches[0]
        annotations[annotation_folder] = annotation_path

    assert len(annotations) == len(annotation_folders)
    return annotations


def create_consensus_step3(image_path, check):
    print("Compute consensus annotations for", image_path)
    annotation_paths = match_annotations(image_path, ANNOTATION_FOLDERS)
    matching_distance = 8
    consensus_annotations, unmatched_annotations = create_consensus_annotations(
        annotation_paths, matching_distance=matching_distance, min_matches_for_consensus=2,
    )
    fname = os.path.basename(image_path)

    prev_consensus = match_annotations(image_path, [CONSENSUS_ANNOTATIONS])[CONSENSUS_ANNOTATIONS]
    prev_consensus = pd.read_csv(prev_consensus)[["axis-0", "axis-1", "axis-2"]]

    if check:
        import napari
        import tifffile

        consensus_annotations = consensus_annotations[["axis-0", "axis-1", "axis-2"]].values
        unmatched_annotators = unmatched_annotations.annotator.values
        unmatched_annotations = unmatched_annotations[["axis-0", "axis-1", "axis-2"]].values

        image = tifffile.imread(image_path)
        v = napari.Viewer()
        v.add_image(image)
        if prev_consensus is not None:
            v.add_points(prev_consensus.values, face_color="gray", name="previous-consensus-annotations")
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
        # Combine consensus and previous annotations.
        consensus_annotations = consensus_annotations[["axis-0", "axis-1", "axis-2"]]
        n_consensus = len(consensus_annotations)
        n_unmatched = len(unmatched_annotations)
        if prev_consensus is not None:
            n_prev = len(prev_consensus)
            print("Number of previous consensus annotations:", n_prev)

        print("Number of new consensus annotations:", n_consensus)
        print("Number of unmatched annotations:", n_unmatched)

        consensus_annotations = pd.concat([consensus_annotations, prev_consensus])

        out_name = fname.replace(".tif", ".csv")
        out_path = os.path.join(OUTPUT_FOLDER, out_name)

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        consensus_annotations.to_csv(out_path, index=False)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.images is None:
        image_paths = sorted(glob(os.path.join(IMAGE_ROOT, "*.tif")))
    else:
        image_paths = args.images

    for image_path in image_paths:
        create_consensus_step3(image_path, args.check)


if __name__ == "__main__":
    main()
