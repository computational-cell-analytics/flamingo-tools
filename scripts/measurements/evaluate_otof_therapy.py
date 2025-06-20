import os
import json

import pandas as pd

from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target

OUTPUT_FOLDER = "./results/otof-measurements"


def check_project(save=False):
    s3 = create_s3_target()

    # content = s3.open(f"{BUCKET_NAME}/project.json", mode="r", encoding="utf-8")
    # x = json.loads(content.read())
    # print(x)
    # return

    cochleae = ["M_AMD_OTOF1_L", "M_AMD_OTOF2_L"]
    ihc_name = "IHC_v2"

    for cochlea in cochleae:
        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        sources = info["sources"]

        source_names = list(sources.keys())
        assert ihc_name in source_names

        # Get the ihc table folder.
        ihc = sources[ihc_name]["segmentation"]
        table_folder = os.path.join(BUCKET_NAME, cochlea, ihc["tableData"]["tsv"]["relativePath"])

        # For debugging.
        # print(s3.ls(table_folder))

        default_table = s3.open(os.path.join(table_folder, "default.tsv"), mode="rb")
        default_table = pd.read_csv(default_table, sep="\t")

        measurement_table = s3.open(os.path.join(table_folder, "Apha_IHC-v2_object-measures.tsv"), mode="rb")
        measurement_table = pd.read_csv(measurement_table, sep="\t")
        if save:
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            measurement_table.to_csv(os.path.join(OUTPUT_FOLDER, f"{cochlea}.csv"), index=False)

        print("Cochlea:", cochlea)
        print("AlphaTag measurements for:", len(measurement_table), "IHCs:")
        print(measurement_table.columns)
        print()


def main():
    check_project(save=True)


if __name__ == "__main__":
    main()
