import argparse
import json
import os
from typing import Optional

import pandas as pd
from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.segmentation.cochlea_mapping import equidistant_centers


def repro_equidistant_centers(
    ddict: dict,
    output_path: str,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    force_overwrite: Optional[bool] = None,
):
    default_cell_type = "ihc"
    default_component_list = [1]
    default_halo_size = [256, 256, 50]
    default_n_blocks = 6

    with open(ddict, 'r') as myfile:
        data = myfile.read()
    param_dicts = json.loads(data)

    out_dict = []

    if os.path.isfile(output_path) and not force_overwrite:
        print(f"Skipping {output_path}. File already exists.")

    for dic in param_dicts:
        cochlea = dic["cochlea"]
        img_channel = dic["image_channel"]
        seg_channel = dic["segmentation_channel"]

        s3_path = os.path.join(f"{cochlea}", "tables", f"{seg_channel}", "default.tsv")
        print(f"Finding equidistant centers for {cochlea}.")

        tsv_path, fs = get_s3_path(s3_path, bucket_name=s3_bucket_name,
                                   service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        with fs.open(tsv_path, 'r') as f:
            table = pd.read_csv(f, sep="\t")

        cell_type = dic["type"] if "type" in dic else default_cell_type
        component_list = dic["component_list"] if "component_list" in dic else default_component_list
        halo_size = dic["halo_size"] if "halo_size" in dic else default_halo_size
        n_blocks = dic["n_blocks"] if "n_blocks" in dic else default_n_blocks

        centers = equidistant_centers(table, component_label=component_list, cell_type=cell_type, n_blocks=n_blocks)
        centers = [[int(c) for c in center] for center in centers]
        ddict = {"cochlea": cochlea}
        ddict["image_channel"] = img_channel
        ddict["crop_centers"] = centers
        ddict["halo_size"] = halo_size
        out_dict.append(ddict)

    with open(output_path, "w") as f:
        json.dump(out_dict, f, indent='\t', separators=(',', ': '))


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract region of interest (ROI) block around center coordinate.")

    parser.add_argument('-i', '--input', type=str, required=True, help="Input JSON dictionary.")
    parser.add_argument('-o', "--output", type=str, required=True, help="Output JSON dictionary.")

    parser.add_argument("--force", action="store_true", help="Forcefully overwrite output.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    repro_equidistant_centers(
        args.input, args.output,
        args.s3_credentials, args.s3_bucket_name, args.s3_service_endpoint,
        args.force,
    )


if __name__ == "__main__":

    main()
