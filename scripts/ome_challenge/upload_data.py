import os
from subprocess import run

ROOT = "/mnt/lustre-emmy-hdd/usr/u12086/data/flamingo"


def upload_data(name):
    data_root = os.path.join(ROOT, "ngff-v3", name)
    assert os.path.exists(data_root), data_root

    bucket_name = "n4bi-goe"

    # Create the bucket.
    cmd = [
        "mc-client", "mb", f"challenge/{bucket_name}/{name}/"
    ]
    run(cmd)

    # Run the copy.
    cmd = [
        "mc-client", "cp", "--recursive",
        f"{data_root}/", f"challenge/{bucket_name}/{name}/"
    ]
    run(cmd)


def main():
    # name = "Platynereis-H2B-TL.ome.zarr"
    # name = "Zebrafish-H2B-short-timelapse.ome.zarr"
    name = "Zebrafish-XSPIM-multiview.ome.zarr"

    upload_data(name)


if __name__ == "__main__":
    main()
