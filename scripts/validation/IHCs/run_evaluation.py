import os
from glob import glob

import pandas as pd
from flamingo_tools.validation import (
    fetch_data_for_evaluation, _parse_annotation_path, compute_scores_for_annotated_slice
)

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationIHCs"
# ANNOTATION_FOLDERS = ["AnnotationsEK", "AnnotationsAMD", "AnnotationsLR"]
# ANNOTATION_FOLDERS = ["Annotations_AMD", "Annotations_LR"]
ANNOTATION_FOLDERS = ["consensus_annotation"]


def run_evaluation(root, annotation_folders, result_file, cache_folder, segmentation_name, exclude):
    results = {
        "annotator": [],
        "cochlea": [],
        "slice": [],
        "tps": [],
        "fps": [],
        "fns": [],
    }

    if cache_folder is not None:
        os.makedirs(cache_folder, exist_ok=True)

    for folder in annotation_folders:
        annotator = "consensus" if folder == "consensus_annotation" else folder[len("Annotations"):]
        annotations = sorted(glob(os.path.join(root, folder, "*.csv")))
        for annotation_path in annotations:
            print(annotation_path)
            cochlea, slice_id = _parse_annotation_path(annotation_path)

            # For the cochlea M_LR_000226_R the actual component is 2, not 1. (Only for IHC_v2).
            component = 2 if ("226_R" in cochlea and segmentation_name == "IHC_v2") else 1
            print("Run evaluation for", annotator, cochlea, "z=", slice_id)
            segmentation, annotations = fetch_data_for_evaluation(
                annotation_path, components_for_postprocessing=[component],
                seg_name=segmentation_name,
                cache_path=None if cache_folder is None else os.path.join(cache_folder, f"{cochlea}_{slice_id}.tif"),
                exclude_zero_synapse_count=exclude,
            )
            scores = compute_scores_for_annotated_slice(segmentation, annotations, matching_tolerance=5)
            results["annotator"].append(annotator)
            results["cochlea"].append(cochlea)
            results["slice"].append(slice_id)
            results["tps"].append(scores["tp"])
            results["fps"].append(scores["fp"])
            results["fns"].append(scores["fn"])

    table = pd.DataFrame(results)
    table.to_csv(result_file, index=False)
    print(table)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=ROOT)
    parser.add_argument("--folders", default=ANNOTATION_FOLDERS)
    parser.add_argument("--result_file", default="results.csv")
    parser.add_argument("--segmentation_name", default="IHC_v4c")
    parser.add_argument("--cache_folder")
    parser.add_argument("--exclude", action="store_true")
    args = parser.parse_args()
    run_evaluation(args.input, args.folders, args.result_file, args.cache_folder, args.segmentation_name, args.exclude)


if __name__ == "__main__":
    main()
