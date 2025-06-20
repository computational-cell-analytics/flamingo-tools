import os
from glob import glob

import pandas as pd
from flamingo_tools.validation import match_detections

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationIHCs"
ANNOTATION_FOLDERS = ["Annotations_AMD", "Annotations_EK", "Annotations_LR"]
CONSENSUS_FOLDER = "consensus_annotation"


def match_annotations(consensus_path):
    annotations = {}
    prefix = os.path.basename(consensus_path).split("_")[:3]
    prefix = "_".join(prefix)

    annotations = {}
    for annotation_folder in ANNOTATION_FOLDERS:
        all_annotations = glob(os.path.join(ROOT, annotation_folder, "*.csv"))
        matches = [ann for ann in all_annotations if os.path.basename(ann).startswith(prefix)]
        assert len(matches) == 1
        annotation_path = matches[0]
        annotations[annotation_folder] = annotation_path

    return annotations


def main():
    consensus_files = sorted(glob(os.path.join(ROOT, CONSENSUS_FOLDER, "*.csv")))

    results = {
        "annotator": [],
        "tps": [],
        "fps": [],
        "fns": [],
    }
    for consensus_file in consensus_files:
        consensus = pd.read_csv(consensus_file)
        consensus = consensus[consensus.annotator == "consensus"][["axis-0", "axis-1", "axis-2"]]

        annotations = match_annotations(consensus_file)
        for name, annotation_path in annotations.items():
            annotation = pd.read_csv(annotation_path)[["axis-0", "axis-1", "axis-2"]]
            tp, _, fp, fn = match_detections(annotation, consensus, max_dist=12.0)
            results["annotator"].append(name)
            results["tps"].append(len(tp))
            results["fps"].append(len(fp))
            results["fns"].append(len(fn))

    results = pd.DataFrame(results)
    print(results)
    results.to_csv("consensus_evaluation.csv", index=False)


if __name__ == "__main__":
    main()
