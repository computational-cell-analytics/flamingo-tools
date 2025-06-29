import argparse

import flamingo_tools.s3_utils as s3_utils
from flamingo_tools.segmentation.synapse_detection import run_prediction


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to image data to be segmented.")
    parser.add_argument("-o", "--output_folder", required=True, help="Path to output folder.")
    parser.add_argument("-m", "--model", required=True,
                        help="Path to synapse detection model.")
    parser.add_argument("-k", "--input_key", default=None,
                        help="The key / internal path to image data.")

    parser.add_argument("--s3", action="store_true", help="Use S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    block_shape = (64, 256, 256)
    halo = (16, 64, 64)

    if args.s3:
        input_path, fs = s3_utils.get_s3_path(args.input, bucket_name=args.s3_bucket_name,
                                              service_endpoint=args.s3_service_endpoint,
                                              credential_file=args.s3_credentials)

    else:
        input_path = args.input

    run_prediction(input_path=input_path, input_key=args.input_key, output_folder=args.output_folder,
                   model_path=args.model, block_shape=block_shape, halo=halo)


if __name__ == "__main__":
    main()
