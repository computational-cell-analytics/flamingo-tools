import os
from glob import glob

from flamingo_tools.validation import create_consensus_annotations

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs"

COLOR = ["blue", "yellow", "orange"]

# First iteration of consensus annotations.
OUTPUT_FOLDER = os.path.join(ROOT, "for_consensus_annotation")
ANNOTATION_FOLDERS = ["AnnotationsAMD", "AnnotationsEK", "AnnotationsLR"]

# Second iteration of consensus annotations for the two images of rescaled cochlea.
# OUTPUT_FOLDER = os.path.join(ROOT, "for_consensus_annotation2")
# ANNOTATION_FOLDERS = [
#     "for_consensus_annotation2/AnnotationsAMD", "for_consensus_annotation2/AnnotationsEK",
#     "for_consensus_annotation2/AnnotationsLR"
# ]


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
    print("Compute consensus annotations for", image_path)
    annotation_paths = match_annotations(image_path)
    matching_distance = 8
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
        # Save the consensus and unmatched annotation.
        consensus_annotations = consensus_annotations[["axis-0", "axis-1", "axis-2"]]
        unmatched_annotations = unmatched_annotations[["axis-0", "axis-1", "axis-2"]]

        consensus_out = os.path.join(OUTPUT_FOLDER, "consensus_annotations")
        os.makedirs(consensus_out, exist_ok=True)

        unmatched_out = os.path.join(OUTPUT_FOLDER, "unmatched_annotations")
        os.makedirs(unmatched_out, exist_ok=True)

        out_name = fname.replace(".tif", ".csv")
        consensus_out_path = os.path.join(consensus_out, out_name)
        unmatched_out_path = os.path.join(unmatched_out, out_name)

        consensus_annotations.to_csv(consensus_out_path, index=False)
        unmatched_annotations.to_csv(unmatched_out_path, index=False)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.images is None:
        image_paths = sorted(glob(os.path.join(OUTPUT_FOLDER, "*.tif")))
    else:
        image_paths = args.images

    for image_path in image_paths:
        consensus_annotations(image_path, args.check)


if __name__ == "__main__":
    main()
