import os

import s3fs
import zarr

from mobie.metadata import add_remote_project_metadata
from tqdm import tqdm

import flamingo_tools.s3_utils as s3_utils

# Using incucyte s3 as a temporary measure.
MOBIE_FOLDER = "/mnt/lustre-emmy-hdd/projects/nim00007/data/moser/lightsheet/mobie"
SERVICE_ENDPOINT = "https://s3.gwdg.de/"
BUCKET_NAME = "incucyte-general/lightsheet"

# FIXME: Complete overhaul with flexible folder, service endpoint, bucket name

# FIXME: access via s3 is not working due to permission issues.
# Maybe this is not working due to bdv fileformat?!
# Make an issue in MoBIE.
def main():
    # remote_metadata()
    s3_utils.upload_data()


def remote_metadata():
    add_remote_project_metadata(MOBIE_FOLDER, BUCKET_NAME, SERVICE_ENDPOINT)


if __name__ == "__main__":
    main()
