import os
from glob import glob
from pathlib import Path

import pandas as pd
from flamingo_tools.validation import match_detections

# The regular root folder.
ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs"
# The root folder for the new annotations for data with scaling issues.
ROOT2 = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs/for_consensus_annotation"  # noqa

ANNOTATION_FOLDERS = ["AnnotationsAMD", "AnnotationsEK", "AnnotationsLR"]
CONSENSUS_FOLDER = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs/final_annotations/final_consensus_annotations"  # noqa


def match_annotations(consensus_path, sample_name):
    annotations = {}
    prefix = os.path.basename(consensus_path).split("_")[:3]
    prefix = "_".join(prefix)

    if sample_name in ("MLR169R_PV_z1913_base_full_rescaled", "MLR169R_PV_z2594_mid_full_rescaled"):
        root = ROOT2
    else:
        root = ROOT

    annotations = {}
    for annotation_folder in ANNOTATION_FOLDERS:
        all_annotations = glob(os.path.join(root, annotation_folder, "*.csv"))
        matches = [
            ann for ann in all_annotations if (os.path.basename(ann).startswith(prefix) and "negative" not in ann)
        ]
        # TODO continue debugging
        if len(matches) != 1:
            breakpoint()
        assert len(matches) == 1
        annotation_path = matches[0]
        annotations[annotation_folder] = annotation_path

    return annotations


def main():
    consensus_files = sorted(glob(os.path.join(CONSENSUS_FOLDER, "*.csv")))
    assert len(consensus_files) > 0

    results = {
        "annotator": [],
        "sample": [],
        "tps": [],
        "fps": [],
        "fns": [],
    }
    for consensus_file in consensus_files:
        consensus = pd.read_csv(consensus_file)
        consensus = consensus[["axis-0", "axis-1", "axis-2"]]
        sample_name = Path(consensus_file).stem

        annotations = match_annotations(consensus_file, sample_name)
        for name, annotation_path in annotations.items():
            annotation = pd.read_csv(annotation_path)[["axis-0", "axis-1", "axis-2"]]
            tp, _, fp, fn = match_detections(annotation, consensus, max_dist=8.0)
            results["annotator"].append(name)
            results["tps"].append(len(tp))
            results["fps"].append(len(fp))
            results["fns"].append(len(fn))
            results["sample"].append(sample_name)

    results = pd.DataFrame(results)
    print(results)
    results.to_csv("results/consensus_evaluation.csv", index=False)


if __name__ == "__main__":
    main()
