import json
import os

import numpy as np
import pandas as pd
from flamingo_tools.s3_utils import create_s3_target, BUCKET_NAME


def open_json(fs, path):
    s3_path = os.path.join(BUCKET_NAME, path)
    with fs.open(s3_path, "r") as f:
        content = json.load(f)
    return content


def open_tsv(fs, path):
    s3_path = os.path.join(BUCKET_NAME, path)
    with fs.open(s3_path, "r") as f:
        table = pd.read_csv(f, sep="\t")
    return table


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cochleae", "-c", nargs="+")
    args = parser.parse_args()

    fs = create_s3_target()
    project_info = open_json(fs, "project.json")

    if args.cochleae is None:
        cochleae = project_info["datasets"]
    else:
        cochleae = args.cochleae

    for dataset in cochleae:
        if dataset not in project_info["datasets"]:
            print("Could not find cochleae", dataset)
            continue
        print(dataset)
        dataset_info = open_json(fs, os.path.join(dataset, "dataset.json"))
        sources = dataset_info["sources"]
        for source, source_info in sources.items():
            if not source.startswith("SGN"):
                continue
            assert "segmentation" in source_info
            source_info = source_info["segmentation"]
            table_path = source_info["tableData"]["tsv"]["relativePath"]
            table = open_tsv(fs, os.path.join(dataset, table_path, "default.tsv"))

            if hasattr(table, "component_labels"):
                component_labels = table.component_labels.values
                remaining_sgns = component_labels[component_labels != 0]
                print(source)
                print(
                    "Number of SGNs (all components)   :", len(remaining_sgns), "/", len(table),
                    "(total number of segmented objects)"
                )
                component_ids, n_per_component = np.unique(
                    remaining_sgns, return_counts=True
                )
                print("Number of SGNs (largest component):", max(n_per_component))
            else:
                print(source)
                print("Number of SGNs (no postprocessing):", len(table))


if __name__ == "__main__":
    main()
