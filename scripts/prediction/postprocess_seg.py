import argparse
import os
import sys

import pandas as pd
import zarr

sys.path.append("../..")

import flamingo_tools.s3_utils as s3_utils

def main():
    from flamingo_tools.segmentation import filter_isolated_objects

    parser = argparse.ArgumentParser(
        description="Script for postprocessing segmentation data in zarr format. Either locally or on an S3 bucket.")

    parser.add_argument("-o", "--output_folder", required=True)

    parser.add_argument("-t", "--tsv", default=None, help="TSV-file in MoBIE format which contains information about the segmentation")
    parser.add_argument('-k', "--input_key", type=str, default="segmentation", help="Input key for data in input file")
    parser.add_argument("--output_key", type=str, default="segmentation_postprocessed", help="Output key for data in input file")

    parser.add_argument("--s3_input", default=None, help="Input file path on S3 bucket")
    parser.add_argument("--s3_credentials", default=None, help="Input file containing S3 credentials. Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported")
    parser.add_argument("--s3_bucket_name", default=None, help="S3 bucket name. Optional if BUCKET_NAME was exported")
    parser.add_argument("--s3_service_endpoint", default=None, help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported")

    parser.add_argument("--min_size", type=int, default=None, help="Minimal number of voxel size for counting object")
    parser.add_argument("--distance_threshold", type=int, default=15, help="Distance in micrometer to check for neighbors")
    parser.add_argument("--neighbor_threshold", type=int, default=5, help="Minimal number of neighbors for filtering")

    args = parser.parse_args()

    seg_path = os.path.join(args.output_folder, "segmentation.zarr")

    tsv_table=None

    if args.s3_input is not None:
        s3_path, fs = s3_utils.get_s3_path(args.s3_input, bucket_name=args.s3_bucket_name, service_endpoint=args.s3_service_endpoint, credential_file=args.s3_credentials)
        with zarr.open(s3_path, mode="r") as f:
            segmentation = f[args.input_key]

        if args.tsv is not None:
            tsv_path, fs = s3_utils.get_s3_path(args.tsv, bucket_name=args.s3_bucket_name, service_endpoint=args.s3_service_endpoint, credential_file=args.s3_credentials)
            with fs.open(tsv_path, 'r') as f:
                tsv_table = pd.read_csv(f, sep="\t")

    else:
        with zarr.open(seg_path, mode="r") as f:
            segmentation = f[args.input_key]

        if args.tsv is not None:
            with open(args.tsv, 'r') as f:
                tsv_table = pd.read_csv(f, sep="\t")

    seg_filtered, n_pre, n_post = filter_isolated_objects(
        segmentation, output_path=seg_path, tsv_table=tsv_table, min_size=args.min_size,
        distance_threshold=args.distance_threshold, neighbor_threshold=args.neighbor_threshold,
        output_key=args.output_key,
        )

    print(f"Number of pre-filtered objects: {n_pre}\nNumber of post-filtered objects: {n_post}")

if __name__ == "__main__":
    main()
