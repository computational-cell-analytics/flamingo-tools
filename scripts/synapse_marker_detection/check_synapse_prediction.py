import os
from glob import glob

import h5py
import imageio.v3 as imageio
import napari
import zarr

from torch_em.util import load_model
from torch_em.util.prediction import predict_with_halo
from train_synapse_detection import get_paths
from tqdm import tqdm

OUTPUT_ROOT = "./predictions"


def run_prediction(val_image):
    model = load_model("./checkpoints/synapse_detection_v1")
    block_shape = (32, 384, 384)
    halo = (8, 64, 64)
    pred = predict_with_halo(val_image, model, [0], block_shape, halo)
    return pred.squeeze()


def require_prediction(image_data, output_path):
    key = "prediction"
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            pred = f[key][:]
    else:
        pred = run_prediction(image_data)
        with h5py.File(output_path, "w") as f:
            f.create_dataset(key, data=pred, compression="gzip")
    return pred


def visualize_results(image_data, pred):
    v = napari.Viewer()
    v.add_image(image_data)
    v.add_image(pred)
    napari.run()


def check_val_image():
    val_paths, _ = get_paths("val")
    val_path = val_paths[0]
    val_image = zarr.open(val_path)["raw"][:]

    os.makedirs(os.path.join(OUTPUT_ROOT, "val"), exist_ok=True)
    output_path = os.path.join(OUTPUT_ROOT, "val", os.path.basename(val_path).replace(".zarr", ".h5"))
    pred = require_prediction(val_image, output_path)

    visualize_results(val_image, pred)


def check_new_images():
    input_root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/test_crops"
    inputs = glob(os.path.join(input_root, "*.tif"))
    output_folder = os.path.join(OUTPUT_ROOT, "new_crops")
    os.makedirs(output_folder, exist_ok=True)
    for path in tqdm(inputs):
        name = os.path.basename(path)
        if name == "M_AMD_58L_avgblendfused_RibB.tif":
            continue
        image_data = imageio.imread(path)
        output_path = os.path.join(output_folder, name.replace(".tif", ".h5"))
        require_prediction(image_data, output_path)


# TODO update to support post-processing and showing annotations for the val data
def main():
    # check_val_image()
    check_new_images()


if __name__ == "__main__":
    main()
