import os
import json

import numpy as np
import pandas as pd

from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target


def check_project():
    s3 = create_s3_target()
    cochleae = ['M_LR_000226_L', 'M_LR_000226_R', 'M_LR_000227_L', 'M_LR_000227_R']

    for cochlea in cochleae:
        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        sources = info["sources"]

        # Load the synapse table.
        syn = sources["synapse_v3"]["spots"]
        rel_path = syn["tableData"]["tsv"]["relativePath"]
        table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")
        syn_table = pd.read_csv(table_content, sep="\t")

        # Load the corresponding ihc table.
        ihc = sources["IHC_v2"]["segmentation"]
        rel_path = ihc["tableData"]["tsv"]["relativePath"]
        table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")
        ihc_table = pd.read_csv(table_content, sep="\t")

        # Keep only the synapses that were matched to a valid IHC.
        component_id = 2 if cochlea == "M_LR_000226_R" else 1
        valid_ihcs = ihc_table.label_id[ihc_table.component_labels == component_id].values

        valid_syn_table = syn_table[syn_table.matched_ihc.isin(valid_ihcs)]
        n_synapses = len(valid_syn_table)

        _, syn_per_ihc = np.unique(valid_syn_table.matched_ihc.values, return_counts=True)

        print("Cochlea:", cochlea)
        print("N-Synapses:", n_synapses)
        print("Average Syn per IHC:", np.mean(syn_per_ihc))
        print("STDEV Syn per IHC:", np.std(syn_per_ihc))
        print()


def main():
    check_project()


if __name__ == "__main__":
    main()
