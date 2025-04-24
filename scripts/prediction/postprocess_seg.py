import argparse
import os

import pandas as pd
import zarr

import flamingo_tools.s3_utils as s3_utils
from flamingo_tools.segmentation import filter_segmentation
from flamingo_tools.segmentation.postprocessing import nearest_neighbor_distance, local_ripleys_k, neighbors_in_radius


# TODO needs updates
def main():

    parser = argparse.ArgumentParser(
        description="Script for postprocessing segmentation data in zarr format. Either locally or on an S3 bucket.")

    parser.add_argument("-o", "--output_folder", type=str, required=True)

    parser.add_argument("-t", "--tsv", type=str, default=None,
                        help="TSV-file in MoBIE format which contains information about segmentation.")
    parser.add_argument('-k', "--input_key", type=str, default="segmentation",
                        help="The key / internal path of the segmentation.")
    parser.add_argument("--output_key", type=str, default="segmentation_postprocessed",
                        help="The key / internal path of the output.")
    parser.add_argument('-r', "--resolution", type=float, default=0.38,
                        help="Resolution of segmentation in micrometer.")

    parser.add_argument("--s3_input", type=str, default=None, help="Input file path on S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    parser.add_argument("--min_size", type=int, default=1000, help="Minimal number of voxel size for counting object")

    parser.add_argument("--n_neighbors", type=int, default=None,
                        help="Value for calculating distance to 'n' nearest neighbors.")

    parser.add_argument("--local_ripley_radius", type=int, default=None,
                        help="Value for radius for calculating local Ripley's K function.")

    parser.add_argument("--r_neighbors", type=int, default=None,
                        help="Value for radius for calculating number of neighbors in range.")

    args = parser.parse_args()

    postprocess_functions = [nearest_neighbor_distance, local_ripleys_k, neighbors_in_radius]
    function_keywords = ["n_neighbors", "radius", "radius"]
    postprocess_options = [args.n_neighbors, args.local_ripley_radius, args.r_neighbors]
    default_thresholds = [15, 20, 20]

    def create_spatial_statistics_dict(functions, keyword, options, threshold):
        spatial_statistics_dict = []
        for f, o, k, t in zip(functions, keyword, options, threshold):
            dic = {"function": f, "keyword": k, "argument": o, "threshold": t}
            spatial_statistics_dict.append(dic)
        return spatial_statistics_dict

    spatial_statistics_dict = create_spatial_statistics_dict(postprocess_functions, postprocess_options,
                                                             function_keywords, default_thresholds)

    if sum(x["argument"] is not None for x in spatial_statistics_dict) == 0:
        raise ValueError("Choose a postprocess function from 'n_neighbors, 'local_ripley_radius', or 'r_neighbors'.")
    elif sum(x["argument"] is not None for x in spatial_statistics_dict) > 1:
        raise ValueError("The script only supports a single postprocess function.")
    else:
        for d in spatial_statistics_dict:
            if d["argument"] is not None:
                spatial_statistics = d["function"]
                spatial_statistics_kwargs = {d["keyword"]: d["argument"]}
                threshold = d["threshold"]

    seg_path = os.path.join(args.output_folder, "segmentation.zarr")

    tsv_table = None

    if args.s3_input is not None:
        s3_path, fs = s3_utils.get_s3_path(args.s3_input, bucket_name=args.s3_bucket_name,
                                           service_endpoint=args.s3_service_endpoint,
                                           credential_file=args.s3_credentials)
        with zarr.open(s3_path, mode="r") as f:
            segmentation = f[args.input_key]

        if args.tsv is not None:
            tsv_path, fs = s3_utils.get_s3_path(args.tsv, bucket_name=args.s3_bucket_name,
                                                service_endpoint=args.s3_service_endpoint,
                                                credential_file=args.s3_credentials)
            with fs.open(tsv_path, 'r') as f:
                tsv_table = pd.read_csv(f, sep="\t")

    else:
        with zarr.open(seg_path, mode="r") as f:
            segmentation = f[args.input_key]

        if args.tsv is not None:
            with open(args.tsv, 'r') as f:
                tsv_table = pd.read_csv(f, sep="\t")

    n_pre, n_post = filter_segmentation(segmentation, output_path=seg_path,
                                        spatial_statistics=spatial_statistics,
                                        threshold=threshold,
                                        min_size=args.min_size, table=tsv_table,
                                        resolution=args.resolution,
                                        output_key=args.output_key, **spatial_statistics_kwargs)

    print(f"Number of pre-filtered objects: {n_pre}\nNumber of post-filtered objects: {n_post}")


if __name__ == "__main__":
    main()
