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


def extract_training_data(imaris_file, output_folder, crop=True, scale=True):
    point_key = "/Scene/Content/Points0/CoordsXYZR"
    with h5py.File(imaris_file, "r") as f:
        if point_key not in f:
            print("Skipping", imaris_file, "due to missing annotations")
            return
        data = f["/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"][:]
        points = f[point_key][:]
        points = points[:, :-1]
        points = points[:, ::-1]

    # TODO crop the data to the original shape.
    # Can we just crop the zero-padding ?!
    if crop:
        crop_box = np.where(data != 0)
        crop_box = tuple(slice(0, int(cb.max() + 1)) for cb in crop_box)
        data = data[crop_box]

    # Scale the points to match the image dimensions.
    voxel_size = get_voxel_size(imaris_file)
    if scale:
        points /= voxel_size[None]

    print(data.shape, voxel_size)

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
def process_training_data_v1():
    files = sorted(glob("./data/synapse_stains/*.ims"))
    for ff in files:
        extract_training_data(ff, output_folder="./training_data")


def process_training_data_v2(visualize=True):
    input_root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/ImageCropsIHC_synapses"

    train_output = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/training_data/v2"  # noqa
    test_output = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/test/v2"  # noqa

    train_folders = ["M78L_IHC-synapse_crops"]
    test_folders = ["M226L_IHC-synapse_crops", "M226R_IHC-synapsecrops"]

    valid_files = [
        "m78l_apexp2718_cr-ctbp2.ims",
        "m226r_apex_p1268_pv-ctbp2.ims",
        "m226r_base_p800_vglut3-ctbp2.ims",
    ]

    for folder in train_folders + test_folders:

        if visualize:
            output_folder = None
        elif folder in train_folders:
            output_folder = train_output
            os.makedirs(output_folder, exist_ok=True)
        else:
            output_folder = test_output
            os.makedirs(output_folder, exist_ok=True)

        imaris_files = sorted(glob(os.path.join(input_root, folder, "*.ims")))
        for imaris_file in imaris_files:
            fname = os.path.basename(imaris_file)
            if fname not in valid_files:
                continue
            print(fname)
            extract_training_data(imaris_file, output_folder, crop=True, scale=True)


def main():
    # process_training_data_v1()
    process_training_data_v2(visualize=False)


if __name__ == "__main__":
    main()
