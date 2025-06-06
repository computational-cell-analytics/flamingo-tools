import argparse
import json
import os
from typing import Optional

import pandas as pd
from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.segmentation.postprocessing import postprocess_sgn_seg


def repro_postprocess_sgn_v1(
    ddict: dict,
    output_dir: str,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    min_size = 1000
    default_threshold_erode = None
    default_min_length = 50
    default_min_edge_distance = 30
    default_iterations_erode = None

    with open(ddict, 'r') as myfile:
        data = myfile.read()
    param_dicts = json.loads(data)

    for dic in param_dicts[2:4]:
        cochlea = dic["cochlea"]
        print(f"Creating components for {cochlea}.")
        suffix = dic["suffix"]
        tsv_path, fs = get_s3_path(dic["s3_path"], bucket_name=s3_bucket_name,
                                   service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        with fs.open(tsv_path, 'r') as f:
            table = pd.read_csv(f, sep="\t")

        threshold_erode = dic["threshold_erode"] if "threshold_erode" in dic else default_threshold_erode
        min_component_length = dic["min_component_length"] if "min_component_length" in dic else default_min_length
        min_edge_distance = dic["min_edge_distance"] if "min_edge_distance" in dic else default_min_edge_distance
        iterations_erode = dic["iterations_erode"] if "iterations_erode" in dic else default_iterations_erode

        print("threshold_erode", threshold_erode)
        print("min_component_length", min_component_length)
        print("min_edge", min_edge_distance)
        print("iterations_erode", iterations_erode)

        tsv_table = postprocess_sgn_seg(table, min_size=min_size,
                                        threshold_erode=threshold_erode,
                                        min_component_length=min_component_length,
                                        min_edge_distance=min_edge_distance,
                                        iterations_erode=iterations_erode)

        largest_comp = len(tsv_table[tsv_table["component_labels"] == 1])
        print(f"Largest component has {largest_comp} SGNs.")

        out_path = os.path.join(output_dir, "".join([cochlea, suffix, ".tsv"]))

        tsv_table.to_csv(out_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract region of interest (ROI) block around center coordinate.")

    parser.add_argument('-i', '--input', type=str, required=True, help="Input JSON dictionary.")
    parser.add_argument('-o', "--output", type=str, required=True, help="Output directory.")

    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    repro_postprocess_sgn_v1(
        args.input, args.output,
        args.s3_credentials, args.s3_bucket_name, args.s3_service_endpoint,
    )


if __name__ == "__main__":

    main()
