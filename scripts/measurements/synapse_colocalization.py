import json
import os

import numpy as np
import pandas as pd

from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target
from flamingo_tools.validation import match_detections

COCHLEA = "M_LR_000215_R"


def _load_table(s3, source, valid_ihcs):
    source_info = source["spots"]
    rel_path = source_info["tableData"]["tsv"]["relativePath"]
    table_content = s3.open(os.path.join(BUCKET_NAME, COCHLEA, rel_path, "default.tsv"), mode="rb")
    synapse_table = pd.read_csv(table_content, sep="\t")
    synapse_table_filtered = synapse_table[synapse_table.matched_ihc.isin(valid_ihcs)]
    return synapse_table_filtered


def _save_ihc_table(table, output_name):
    ihc_ids, syn_per_ihc = np.unique(table.matched_ihc.values, return_counts=True)
    ihc_ids = ihc_ids.astype("int")
    ihc_to_count = {ihc_id: count for ihc_id, count in zip(ihc_ids, syn_per_ihc)}
    ihc_count_table = pd.DataFrame({
        "label_id": list(ihc_to_count.keys()), "synapse_count": list(ihc_to_count.values())
    })
    output_path = os.path.join("ihc_counts", f"ihc_count_{COCHLEA}_{output_name}.tsv")
    ihc_count_table.to_csv(output_path, sep="\t", index=False)


def _run_colocalization(riba_table, ctbp2_table, max_dist=2.0):
    coords_riba = riba_table[["z", "y", "x"]].values
    coords_ctbp2 = ctbp2_table[["z", "y", "x"]].values

    matches_riba, matches_ctbp2, unmatched_riba, unmatched_ctbp2 = match_detections(
        coords_riba, coords_ctbp2, max_dist=max_dist,
    )
    assert len(matches_riba) == len(matches_ctbp2)

    # For quick visualization
    if False:
        matched_coords = coords_riba[matches_riba]

        import napari
        v = napari.Viewer()
        v.add_points(coords_riba, name="RibA", face_color="orange")
        v.add_points(coords_ctbp2, name="CTBP2")
        v.add_points(matched_coords, name="Coloc", face_color="green")
        napari.run()

    return matches_riba, unmatched_riba, unmatched_ctbp2


def check_and_filter_synapses():
    name_ctbp2 = "CTBP2_synapse_v3_ihc_v4b"
    name_riba = "RibA_synapse_v3_ihc_v4b"

    s3 = create_s3_target()
    content = s3.open(f"{BUCKET_NAME}/{COCHLEA}/dataset.json", mode="r", encoding="utf-8")
    info = json.loads(content.read())
    sources = info["sources"]

    # TODO load from S3 instead
    ihc_labels = pd.read_csv("./ihc_counts/ihc-annotation.tsv", sep="\t")
    valid_ihcs = ihc_labels.label_id[ihc_labels.ihc == "is_ihc"].values

    riba_table = _load_table(s3, sources[name_riba], valid_ihcs)
    ctbp2_table = _load_table(s3, sources[name_ctbp2], valid_ihcs)

    # Save the single synapse marker tables.
    _save_ihc_table(riba_table, "RibA")
    _save_ihc_table(ctbp2_table, "CTBP2")

    # Run co-localization, analyze it and save the table.
    matches_riba, unmatched_riba, unmatched_ctbp2 = _run_colocalization(riba_table, ctbp2_table)

    n_matched = len(matches_riba)
    print("Number of IHCs:", len(valid_ihcs))
    print("Number of matched synapses:", n_matched)
    print()

    n_ctbp2 = n_matched + len(unmatched_ctbp2)
    n_riba = n_matched + len(unmatched_riba)
    print("Number and percentage of matched synapses for markers:")
    print("CTBP2:", n_matched, "/", n_ctbp2, f"({float(n_matched) / n_ctbp2 * 100}% matched)")
    print("RibA :", n_matched, "/", n_riba, f"({float(n_matched) / n_riba * 100}% matched)")

    coloc_table = riba_table.iloc[matches_riba]
    _save_ihc_table(coloc_table, "coloc")


def main():
    check_and_filter_synapses()


if __name__ == "__main__":
    main()
