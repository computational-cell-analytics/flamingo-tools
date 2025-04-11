import argparse

import pandas as pd

import flamingo_tools.segmentation.postprocessing as postprocessing
import flamingo_tools.s3_utils as s3_utils

def main(
    in_path, out_path, n_neighbors=None,
    s3=False, s3_credentials=None, s3_bucket_name=None, s3_service_endpoint=None,
    ):
    """

    :param str input_file: Path to table in TSV format
    :param str out_path: Path to save output
    :param bool s3: Flag for using an S3 bucket
    :param str s3_credentials: Path to file containing S3 credentials
    :param str s3_bucket_name: S3 bucket name. Optional if BUCKET_NAME has been exported
    :param str s3_service_endpoint: S3 service endpoint. Optional if SERVICE_ENDPOINT has been exported
    """
    if s3:
        tsv_path, fs = s3_utils.get_s3_path(in_path, bucket_name=s3_bucket_name, service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        with fs.open(tsv_path, 'r') as f:
            tsv_table = pd.read_csv(f, sep="\t")
    else:
        with open(in_path, 'r') as f:
            tsv_table = pd.read_csv(f, sep="\t")

    if n_neighbors is not None:
        nn_list = [int(n) for n in n_neighbors.split(",")]
        for n_neighbor in nn_list:
            if n_neighbor >= len(tsv_table):
                raise ValueError(f"Number of neighbors: {n_neighbor} exceeds number of elements in dataframe: {len(tsv_table)}.")

            _ = postprocessing.distance_nearest_neighbors(tsv_table=tsv_table, n_neighbors=n_neighbor, expand_table=True)

    tsv_table.to_csv(out_path, sep="\t")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for expanding the segmentation table of MoBIE with additonal parameters. Either locally or on an S3 bucket.")

    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)

    parser.add_argument("--n_neighbors", default=None, help="Value(s) for number of nearest neighbors in format 'n1,n2,...,nx'. New columns contain the average distance to nearest neighbors.")

    parser.add_argument("--s3", action="store_true", help="Flag for using S3 bucket")
    parser.add_argument("--s3_credentials", default=None, help="Input file containing S3 credentials. Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported")
    parser.add_argument("--s3_bucket_name", default=None, help="S3 bucket name. Optional if BUCKET_NAME was exported")
    parser.add_argument("--s3_service_endpoint", default=None, help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported")

    args = parser.parse_args()

    main(
        args.input, args.output, args.n_neighbors,
        args.s3, args.s3_credentials, args.s3_bucket_name, args.s3_service_endpoint,
    )
