import os
import numpy as np
import pandas as pd

ROOT = "/mnt/lustre-emmy-hdd/usr/u12086/data/flamingo/ngff-v3"
URL_ROOT = "https://radosgw.public.os.wwu.de/n4bi-goe"


def get_directory_size(directory):
    total_size = 0
    # Walk through all subdirectories and files
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Only add file size if it is a file (skip if it's a broken symlink)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)

    size_rd = np.round(total_size / 1e9, 2)
    size_rd = f"{size_rd} GB"
    return total_size, size_rd


names = [
    "Platynereis-H2B-TL.ome.zarr",
    "Zebrafish-H2B-short-timelapse.ome.zarr",
    "Zebrafish-XSPIM-multiview.ome.zarr",
]

urls = []
written = []
written_human_readable = []

for name in names:
    url = f"{URL_ROOT}/{name}"
    urls.append(url)
    folder = os.path.join(ROOT, name)
    size, size_rd = get_directory_size(folder)
    written.append(size)
    written_human_readable.append(size_rd)

df = {
    "url": urls, "written": written, "written_human_readable": written_human_readable,
}

df = pd.DataFrame(df)
df.to_csv("flamingo.csv", index=False)
