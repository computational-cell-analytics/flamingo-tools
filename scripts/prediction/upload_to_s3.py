import os

import s3fs

from mobie.metadata import add_remote_project_metadata
from tqdm import tqdm

# Using incucyte s3 as a temporary measure.
MOBIE_FOLDER = "/mnt/lustre-emmy-hdd/projects/nim00007/data/moser/lightsheet/mobie"
SERVICE_ENDPOINT = "https://s3.gwdg.de/"
BUCKET_NAME = "incucyte-general/lightsheet"

# For MoBIE:
# https://s3.gwdg.de/incucyte-general/lightsheet


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


# FIXME: access via s3 is not working due to permission issues.
# Maybe this is not working due to bdv fileformat?!
# Make an issue in MoBIE.
def main():
    # remote_metadata()
    upload_data()


if __name__ == "__main__":
    main()
