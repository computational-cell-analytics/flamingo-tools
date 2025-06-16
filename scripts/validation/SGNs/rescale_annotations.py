import os
import shutil
from glob import glob

import numpy as np
import pandas as pd
import tifffile
import zarr

from flamingo_tools.s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT


def get_scale_factor():
    original_path = "/mnt/ceph-hdd/cold/nim00007/cochlea-lightsheet/M_LR_000169_R/MLR000169R_PV.tif"
    original_shape = tifffile.memmap(original_path).shape

    cochlea = "M_LR_000169_R"
    internal_path = os.path.join(cochlea, "images",  "ome-zarr", "SGN_v2.ome.zarr")
    s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)

    input_key = "s0"
    with zarr.open(s3_store, mode="r") as f:
        new_shape = f[input_key].shape

    scale_factor = tuple(
        float(nsh) / float(osh) for nsh, osh in zip(new_shape, original_shape)
    )
    return scale_factor


def rescale_annotations(input_path, scale_factor, bkp_folder):
    annotations = pd.read_csv(input_path)

    annotations_rescaled = annotations.copy()
    annotations_rescaled["axis-1"] = annotations["axis-1"] * scale_factor[1]
    annotations_rescaled["axis-2"] = annotations["axis-2"] * scale_factor[2]

    fname = os.path.basename(input_path)
    name_components = fname.split("_")
    z = int(name_components[2][1:])
    new_z = int(np.round(z * scale_factor[0]))

    name_components[2] = f"z{new_z}"
    name_components = name_components[:-1] + ["rescaled"] + name_components[-1:]
    new_fname = "_".join(name_components)

    input_folder = os.path.split(input_path)[0]
    out_path = os.path.join(input_folder, new_fname)
    bkp_path = os.path.join(bkp_folder, fname)

    # print(input_path)
    # print(out_path)
    # print(bkp_path)
    # print()
    # return

    shutil.move(input_path, bkp_path)
    annotations_rescaled.to_csv(out_path, index=False)


def main():
    # scale_factor = get_scale_factor()
    # print(scale_factor)
    scale_factor = (2.6314,) * 3

    root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs"
    annotation_folders = ["AnnotationsEK", "AnnotationsAMD", "AnnotationsLR"]
    for folder in annotation_folders:
        bkp_folder = os.path.join(root, folder, "rescaled_bkp")
        os.makedirs(bkp_folder, exist_ok=True)

        files = glob(os.path.join(root, folder, "*.csv"))
        for annotation_file in files:
            fname = os.path.basename(annotation_file)
            if not fname.startswith(("MLR169R_PV_z722", "MLR169R_PV_z979")):
                continue
            print("Rescaling", annotation_file)
            rescale_annotations(annotation_file, scale_factor, bkp_folder)


# Rescale the point annotations for the cochlea MLR169R, which was
# annotated at the original scale, but then rescaled for segmentation.
if __name__ == "__main__":
    main()
