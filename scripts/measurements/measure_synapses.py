import os
import json

import numpy as np
import pandas as pd

from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target

OUTPUT_FOLDER = "./ihc_counts"


def check_project(plot=False, save_ihc_table=False, max_dist=None):
    s3 = create_s3_target()
    cochleae = ['M_LR_000226_L', 'M_LR_000226_R', 'M_LR_000227_L', 'M_LR_000227_R']
    synapse_table_name = "synapse_v3_ihc_v4"
    ihc_table_name = "IHC_v4"

    results = {}
    for cochlea in cochleae:
        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        sources = info["sources"]

        # Load the synapse table.
        syn = sources[synapse_table_name]["spots"]
        rel_path = syn["tableData"]["tsv"]["relativePath"]
        table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")
        syn_table = pd.read_csv(table_content, sep="\t")
        if max_dist is None:
            max_dist = syn_table.distance_to_ihc.max()
        else:
            syn_table = syn_table[syn_table.distance_to_ihc <= max_dist]

        # Load the corresponding ihc table.
        ihc = sources[ihc_table_name]["segmentation"]
        rel_path = ihc["tableData"]["tsv"]["relativePath"]
        table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")
        ihc_table = pd.read_csv(table_content, sep="\t")

        # Keep only the synapses that were matched to a valid IHC.
        component_id = 1
        valid_ihcs = ihc_table.label_id[ihc_table.component_labels == component_id].values.astype("int")

        valid_syn_table = syn_table[syn_table.matched_ihc.isin(valid_ihcs)]
        n_synapses = len(valid_syn_table)

        ihc_ids, syn_per_ihc = np.unique(valid_syn_table.matched_ihc.values, return_counts=True)
        ihc_ids = ihc_ids.astype("int")
        results[cochlea] = syn_per_ihc

        print("Cochlea:", cochlea)
        print("N-Synapses:", n_synapses)
        print("Average Syn per IHC:", np.mean(syn_per_ihc))
        print("STDEV Syn per IHC:", np.std(syn_per_ihc))
        print("@ max dist:", max_dist)
        print()

        if save_ihc_table:
            ihc_to_count = {ihc_id: count for ihc_id, count in zip(ihc_ids, syn_per_ihc)}
            unmatched_ihcs = np.setdiff1d(valid_ihcs, ihc_ids)
            ihc_to_count.update({ihc_id: 0 for ihc_id in unmatched_ihcs})
            ihc_count_table = pd.DataFrame({
                "label_id": list(ihc_to_count.keys()),
                "synapse_count": list(ihc_to_count.values()),
                "snyapse_table": [synapse_table_name for _ in list(ihc_to_count.values())],
                "ihc_table": [ihc_table_name for _ in list(ihc_to_count.values())],
                "max_dist": [max_dist for _ in list(ihc_to_count.values())],
            })
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            output_path = os.path.join(OUTPUT_FOLDER, f"ihc_count_{cochlea}.tsv")
            ihc_count_table.to_csv(output_path, sep="\t", index=False)

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        cap = 30

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
        for i, (cochlea, res) in enumerate(results.items()):
            sns.histplot(data=np.clip(res, 0, cap), bins=16, ax=axes[i])
            axes[i].set_title(cochlea)

        fig.suptitle(f"Ribbon Synapses per IHC @ {np.round(max_dist)} micron")

        plt.tight_layout()
        plt.show()


def main():
    check_project(plot=False, save_ihc_table=True, max_dist=3)


if __name__ == "__main__":
    main()
