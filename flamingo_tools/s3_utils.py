import os

import s3fs
import zarr

from mobie.metadata import add_remote_project_metadata
from tqdm import tqdm

# Using incucyte s3 as a temporary measure.
MOBIE_FOLDER = "/mnt/lustre-emmy-hdd/projects/nim00007/data/moser/lightsheet/mobie"
SERVICE_ENDPOINT = "https://s3.gwdg.de/"
BUCKET_NAME = "incucyte-general/lightsheet"

# For MoBIE:
# https://s3.gwdg.de/incucyte-general/lightsheet

def check_s3_credentials(bucket_name, service_endpoint, credentials):
    """
    Check if S3 parameter and credentials were set either as a function input or were exported as environment variables.
    """
    if bucket_name is None:
        bucket_name = os.getenv('BUCKET_NAME')
        if bucket_name is None:
            raise ValueError("Provide a bucket name for accessing S3 data.\nEither by using an optional argument or exporting an environment variable:\n--s3_bucket_name <bucket_name>\nexport BUCKET_NAME=<bucket_name>")

    if service_endpoint is None:
        service_endpoint = os.getenv('SERVICE_ENDPOINT')
        if service_endpoint is None:
            raise ValueError("Provide a service endpoint for accessing S3 data.\nEither by using an optional argument or exporting an environment variable:\n--s3_service_endpoint <endpoint>\nexport SERVICE_ENDPOINT=<endpoint>")

    if credentials is None:
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        if access_key is None:
            raise ValueError("Either provide a credential file as an optional argument or export an access key as an environment variable:\nexport AWS_ACCESS_KEY_ID=<access_key>")
        if secret_key is None:
            raise ValueError("Either provide a credential file as an optional argument or export a secret access key as an environment variable:\nexport AWS_SECRET_ACCESS_KEY=<secret_key>")

    return bucket_name, service_endpoint, credentials


def get_s3_path(
    input_path,
    bucket_name, service_endpoint,
    credential_file=None,
):
    """
    Get S3 path for a file or folder and file system based on S3 parameters and credentials.
    """
    fs = create_s3_target(url=service_endpoint, anon=False, credential_file=credential_file)

    zarr_path=f"{bucket_name}/{input_path}"

    if not fs.exists(zarr_path):
        print(f"Error: S3 path {zarr_path} does not exist!")

    s3_path = zarr.storage.FSStore(zarr_path, fs=fs)

    return s3_path, fs


def read_s3_credentials(credential_file):
    key, secret = None, None
    with open(credential_file) as f:
        for line in f:
            if line.startswith("aws_access_key_id"):
                key = line.rstrip("\n").strip().split(" ")[-1]
            if line.startswith("aws_secret_access_key"):
                secret = line.rstrip("\n").strip().split(" ")[-1]
    if key is None or secret is None:
        raise ValueError(f"Invalid credential file {credential_file}")
    return key, secret


def create_s3_target(url, anon=False, credential_file=None):
    """
    Create file system for S3 bucket based on a service endpoint and an optional credential file.
    If the credential file is not provided, the s3fs.S3FileSystem function checks the environment variables
    AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.
    """
    client_kwargs = {"endpoint_url": url}
    if credential_file is not None:
        key, secret = read_s3_credentials(credential_file)
        fs = s3fs.S3FileSystem(key=key, secret=secret, client_kwargs=client_kwargs)
    else:
        fs = s3fs.S3FileSystem(anon=anon, client_kwargs=client_kwargs)
    return fs


def remote_metadata():
    add_remote_project_metadata(MOBIE_FOLDER, BUCKET_NAME, SERVICE_ENDPOINT)


def upload_data():
    target = create_s3_target(
        SERVICE_ENDPOINT,
        credential_file="./credentials.incucyte"
    )
    to_upload = []
    for root, dirs, files in os.walk(MOBIE_FOLDER):
        dirs.sort()
        for ff in files:
            if ff.endswith(".xml"):
                to_upload.append(os.path.join(root, ff))

    print("Uploading", len(to_upload), "files to")

    for path in tqdm(to_upload):
        rel_path = os.path.relpath(path, MOBIE_FOLDER)
        target.put(
            path, os.path.join(BUCKET_NAME, rel_path)
        )