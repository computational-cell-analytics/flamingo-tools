import os
from glob import glob
from pathlib import Path

import h5py
import napari
import numpy as np
import pandas as pd
import zarr


def get_voxel_size(imaris_file):
    with h5py.File(imaris_file, "r") as f:
        info = f["/DataSetInfo/Image"]
        ext = [[float(b"".join(info.attrs[f"ExtMin{i}"]).decode()),
                float(b"".join(info.attrs[f"ExtMax{i}"]).decode())] for i in range(3)]
        size = [int(b"".join(info.attrs[dim]).decode()) for dim in ["X", "Y", "Z"]]
        vsize = np.array([(max_-min_)/s for (min_, max_), s in zip(ext, size)])
    return vsize


def extract_training_data(imaris_file, output_folder):
    with h5py.File(imaris_file, "r") as f:
        data = f["/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"][:]
        points = f["/Scene/Content/Points0/CoordsXYZR"][:]
        points = points[:, :-1]
        points = points[:, ::-1]

    # TODO crop the data to the original shape.
    # Can we just crop the zero-padding ?!
    crop_box = np.where(data != 0)
    crop_box = tuple(slice(0, int(cb.max() + 1)) for cb in crop_box)
    data = data[crop_box]
    print(data.shape)

    # Scale the points to match the image dimensions.
    voxel_size = get_voxel_size(imaris_file)
    points /= voxel_size[None]

    if output_folder is None:
        v = napari.Viewer()
        v.add_image(data)
        v.add_points(points)
        v.title = os.path.basename(imaris_file)
        napari.run()
    else:
        image_folder = os.path.join(output_folder, "images")
        os.makedirs(image_folder, exist_ok=True)

        label_folder = os.path.join(output_folder, "labels")
        os.makedirs(label_folder, exist_ok=True)

        fname = Path(imaris_file).stem
        image_file = os.path.join(image_folder, f"{fname}.zarr")
        label_file = os.path.join(label_folder, f"{fname}.csv")

        coords = pd.DataFrame(points, columns=["axis-0", "axis-1", "axis-2"])
        coords.to_csv(label_file, index=False)

        f = zarr.open(image_file, "a")
        f.create_dataset("raw", data=data)


# Files that look good for training:
# - 4.1L_apex_IHCribboncount_Z.ims
# - 4.1L_base_IHCribbons_Z.ims
# - 4.1L_mid_IHCribboncount_Z.ims
# - 4.2R_apex_IHCribboncount_Z.ims
# - 4.2R_apex_IHCribboncount_Z.ims
# - 6.2R_apex_IHCribboncount_Z.ims  (very small crop)
# - 6.2R_base_IHCribbons_Z.ims
def main():
    files = sorted(glob("./data/synapse_stains/*.ims"))
    for ff in files:
        extract_training_data(ff, output_folder="./training_data")


if __name__ == "__main__":
    main()
