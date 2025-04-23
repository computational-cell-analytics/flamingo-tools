"""This file contains utility functions for processing data located on an S3 storage.
The upload of data to the storage system should be performed with 'rclone'.
"""
import os
from typing import Optional, Tuple

import s3fs
import zarr


# Dedicated bucket for cochlea lightsheet project
MOBIE_FOLDER = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"
SERVICE_ENDPOINT = "https://s3.fs.gwdg.de/"
BUCKET_NAME = "cochlea-lightsheet"

DEFAULT_CREDENTIALS = os.path.expanduser("~/.aws/credentials")


def check_s3_credentials(
    bucket_name: Optional[str], service_endpoint: Optional[str], credential_file: Optional[str]
) -> Tuple[str, str, str]:
    """Check if S3 parameter and credentials were set as input arguments, as environment variables, or as globals.

    Args:
        bucket_name: S3 bucket name
        service_endpoint: S3 service endpoint
        credential_file: Credential file containing access key and secret key

    Returns:
        bucket_name
        service_endpoint
        credential_file
    """
    if bucket_name is None:
        bucket_name = os.getenv("BUCKET_NAME")
        if bucket_name is None:
            if BUCKET_NAME in globals():
                bucket_name = BUCKET_NAME
            else:
                raise ValueError(
                    "Provide a bucket name for accessing S3 data.\n"
                    "Either by using an optional argument or exporting an environment variable:\n"
                    "--s3_bucket_name <bucket_name>\n"
                    "export BUCKET_NAME=<bucket_name>"
                )

    if service_endpoint is None:
        service_endpoint = os.getenv("SERVICE_ENDPOINT")
        if service_endpoint is None:
            if SERVICE_ENDPOINT in globals():
                service_endpoint = SERVICE_ENDPOINT
            else:
                raise ValueError(
                    "Provide a service endpoint for accessing S3 data.\n"
                    "Either by using an optional argument or exporting an environment variable:\n"
                    "--s3_service_endpoint <endpoint>\n"
                    "export SERVICE_ENDPOINT=<endpoint>")

    if credential_file is None:
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        # check for default credentials if no credential_file is provided
        if access_key is None:
            if os.path.isfile(DEFAULT_CREDENTIALS):
                access_key, _ = read_s3_credentials(credential_file=DEFAULT_CREDENTIALS)
            else:
                raise ValueError(
                    "Either provide a credential file as an optional argument,"
                    f" have credentials at '{DEFAULT_CREDENTIALS}',"
                    " or export an access key as an environment variable:\n"
                    "export AWS_ACCESS_KEY_ID=<access_key>")
        if secret_key is None:
            # check for default credentials
            if os.path.isfile(DEFAULT_CREDENTIALS):
                _, secret_key = read_s3_credentials(credential_file=DEFAULT_CREDENTIALS)
            else:
                raise ValueError(
                    "Either provide a credential file as an optional argument,"
                    f" have credentials at '{DEFAULT_CREDENTIALS}',"
                    " or export a secret access key as an environment variable:\n"
                    "export AWS_SECRET_ACCESS_KEY=<secret_key>")

    else:
        # check validity of credential file
        _, _ = read_s3_credentials(credential_file=credential_file)

    return bucket_name, service_endpoint, credential_file


def get_s3_path(
    input_path: str,
    bucket_name: Optional[str] = None,
    service_endpoint: Optional[str] = None,
    credential_file: Optional[str] = None,
) -> Tuple[zarr.storage.FSStore, s3fs.core.S3FileSystem]:
    """Get S3 path for a file or folder and file system based on S3 parameters and credentials.

    Args:
        input_path: Inputh path in S3 bucket
        bucket_name: S3 bucket name
        service_endpoint: S3 service endpoint
        credential_file: Credential file containing access key and secret key

    Returns:
        s3_path
        s3_filesystem
    """
    bucket_name, service_endpoint, credential_file = check_s3_credentials(
        bucket_name, service_endpoint, credential_file
    )

    s3_filesystem = create_s3_target(url=service_endpoint, anon=False, credential_file=credential_file)

    zarr_path = f"{bucket_name}/{input_path}"

    if not s3_filesystem.exists(zarr_path):
        print(f"Error: S3 path {zarr_path} does not exist!")

    s3_path = zarr.storage.FSStore(zarr_path, fs=s3_filesystem)

    return s3_path, s3_filesystem


def read_s3_credentials(credential_file: str) -> Tuple[str, str]:
    """Read access key amd secret key from credential file.

    Args:
        credential_file: File path to credentials

    Returns:
        access_key
        secret_key
    """
    access_key, secret_key = None, None
    with open(credential_file) as f:
        for line in f:
            if line.startswith("aws_access_key_id"):
                access_key = line.rstrip("\n").strip().split(" ")[-1]
            if line.startswith("aws_secret_access_key"):
                secret_key = line.rstrip("\n").strip().split(" ")[-1]
    if access_key is None or secret_key is None:
        raise ValueError(f"Invalid credential file {credential_file}")
    return access_key, secret_key


def create_s3_target(
    url: Optional[str] = None,
    anon: Optional[str] = False,
    credential_file: Optional[str] = None,
) -> s3fs.core.S3FileSystem:
    """Create file system for S3 bucket based on a service endpoint and an optional credential file.
    If the credential file is not provided, the s3fs.S3FileSystem function checks the environment variables
    AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.

    Args:
        url: Service endpoint for S3 bucket
        anon: Option for anon argument of S3FileSystem
        credential_file: File path to credentials

    Returns:
        s3_filesystem
    """
    client_kwargs = {"endpoint_url": SERVICE_ENDPOINT if url is None else url}
    if credential_file is not None:
        key, secret = read_s3_credentials(credential_file)
        s3_filesystem = s3fs.S3FileSystem(key=key, secret=secret, client_kwargs=client_kwargs)
    else:
        s3_filesystem = s3fs.S3FileSystem(anon=anon, client_kwargs=client_kwargs)
    return s3_filesystem
