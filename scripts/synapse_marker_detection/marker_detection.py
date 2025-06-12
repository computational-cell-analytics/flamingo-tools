import argparse

import flamingo_tools.s3_utils as s3_utils
from flamingo_tools.segmentation.synapse_detection import marker_detection


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output_folder", required=True, help="Path to output folder.")
    parser.add_argument("-s", "--mask", required=True, help="Path to IHC segmentation used for masking.")
    parser.add_argument("-m", "--model", required=True, help="Path to synapse detection model.")
    parser.add_argument("-k", "--input_key", default=None,
                        help="Input key for image data and mask data for marker detection.")
    parser.add_argument("-d", "--max_distance", default=20,
                        help="The maximal distance for a valid match of synapse markers to IHCs.")

    parser.add_argument("--s3", action="store_true", help="Use S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    if args.s3:
        input_path, fs = s3_utils.get_s3_path(args.input, bucket_name=args.s3_bucket_name,
                                              service_endpoint=args.s3_service_endpoint,
                                              credential_file=args.s3_credentials)

        mask_path, fs = s3_utils.get_s3_path(args.mask, bucket_name=args.s3_bucket_name,
                                             service_endpoint=args.s3_service_endpoint,
                                             credential_file=args.s3_credentials)
    else:
        input_path = args.input
        mask_path = args.mask

    marker_detection(input_path=input_path, input_key=args.input_key, mask_path=mask_path,
                     output_folder=args.output_folder, model_path=args.model)


if __name__ == "__main__":
    main()
