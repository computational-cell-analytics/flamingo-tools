import os
import json

import pandas as pd

from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target

OUTPUT_FOLDER = "./results/sgn-measurements"


def check_project(save=False):
    s3 = create_s3_target()

    content = s3.open(f"{BUCKET_NAME}/project.json", mode="r", encoding="utf-8")
    project_info = json.loads(content.read())

    cochleae = [
        "M_LR_000144_L", "M_LR_000145_L",
        "M_LR_000151_R", "M_LR_000155_L",
    ]

    sgn_name = "SGN_resized_v2"
    for cochlea in cochleae:
        assert cochlea in project_info["datasets"]

        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        sources = info["sources"]

        source_names = list(sources.keys())
        if sgn_name not in source_names:
            continue

        # Get the ihc table folder.
        ihc = sources[sgn_name]["segmentation"]
        table_folder = os.path.join(BUCKET_NAME, cochlea, ihc["tableData"]["tsv"]["relativePath"])

        # For debugging.
        x = s3.ls(table_folder)
        if len(x) == 1:
            continue

        default_table = s3.open(os.path.join(table_folder, "default.tsv"), mode="rb")
        default_table = pd.read_csv(default_table, sep="\t")

        measurement_table = s3.open(
            os.path.join(table_folder, "GFP-resized_SGN-resized-v2_object-measures.tsv"), mode="rb"
        )
        measurement_table = pd.read_csv(measurement_table, sep="\t")

        if save:
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            measurement_table.to_csv(os.path.join(OUTPUT_FOLDER, f"{cochlea}.csv"), index=False)

        print("Cochlea:", cochlea)
        print("GFP measurements for:", len(measurement_table), "SGNs:")
        print(measurement_table.columns)
        print()


def main():
    check_project(save=True)


if __name__ == "__main__":
    main()
