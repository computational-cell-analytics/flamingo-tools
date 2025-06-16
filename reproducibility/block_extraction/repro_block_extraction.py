import argparse
import json
import os
from typing import Optional

from flamingo_tools.extract_block_util import extract_block


def repro_block_extraction(
    ddict: dict,
    output_dir: str,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    s3_flag = True
    tif_flag = True
    input_key = "s0"

    with open(ddict, 'r') as myfile:
        data = myfile.read()
    param_dicts = json.loads(data)

    for dic in param_dicts:
        if "image_channel" in dic:
            image_channels = dic["image_channel"] if isinstance(dic["image_channel"], list) else [dic["image_channel"]]
            for image_channel in image_channels:
                input_path = os.path.join(dic["cochlea"], "images", "ome-zarr", image_channel + ".ome.zarr")
                roi_halo = dic["halo_size"]
                crop_centers = dic["crop_centers"]
                for coord in crop_centers:
                    extract_block(input_path, coord, output_dir, input_key=input_key, roi_halo=roi_halo, tif=tif_flag,
                                  s3=s3_flag, s3_credentials=s3_credentials, s3_bucket_name=s3_bucket_name,
                                  s3_service_endpoint=s3_service_endpoint)


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

    repro_block_extraction(
        args.input, args.output,
        args.s3_credentials, args.s3_bucket_name, args.s3_service_endpoint,
    )


if __name__ == "__main__":

    main()
