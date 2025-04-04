import os
from glob import glob
from pathlib import Path

import h5py
import imageio.v3 as imageio
import napari
import numpy as np
import pandas as pd
import zarr

# from skimage.feature import blob_dog
from skimage.feature import peak_local_max
from torch_em.util import load_model
from torch_em.util.prediction import predict_with_halo
from train_synapse_detection import get_paths
from tqdm import tqdm

# INPUT_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/test_crops"
INPUT_ROOT = "./data/test_crops"
OUTPUT_ROOT = "./predictions"
DETECTION_OUT_ROOT = "./detections"


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


def run_postprocessing(pred):
    # print("Running local max ...")
    # coords = blob_dog(pred)
    coords = peak_local_max(pred, min_distance=2, threshold_abs=0.5)
    # print("... done")
    return coords


def visualize_results(image_data, pred, coords=None, val_coords=None, title=None):
    v = napari.Viewer()
    v.add_image(image_data)
    pred = pred.clip(0, pred.max())
    v.add_image(pred)
    if coords is not None:
        v.add_points(coords, name="predicted_synapses", face_color="yellow")
    if val_coords is not None:
        v.add_points(val_coords, face_color="green", name="synapse_annotations")
    if title is not None:
        v.title = title
    napari.run()


def check_val_image():
    val_paths, _ = get_paths("val")
    val_path = val_paths[0]
    val_image = zarr.open(val_path)["raw"][:]

    os.makedirs(os.path.join(OUTPUT_ROOT, "val"), exist_ok=True)
    output_path = os.path.join(OUTPUT_ROOT, "val", os.path.basename(val_path).replace(".zarr", ".h5"))
    pred = require_prediction(val_image, output_path)

    visualize_results(val_image, pred)


def check_new_images(view=False, save_detection=False):
    inputs = glob(os.path.join(INPUT_ROOT, "*.tif"))
    output_folder = os.path.join(OUTPUT_ROOT, "new_crops")
    os.makedirs(output_folder, exist_ok=True)
    for path in tqdm(inputs):
        print(path)
        name = os.path.basename(path)
        if name == "M_AMD_58L_avgblendfused_RibB.tif":
            continue
        image_data = imageio.imread(path)
        output_path = os.path.join(output_folder, name.replace(".tif", ".h5"))
        # if not os.path.exists(output_path):
        #     continue
        pred = require_prediction(image_data, output_path)
        if view or save_detection:
            coords = run_postprocessing(pred)
        if view:
            print("Number of synapses:", len(coords))
            visualize_results(image_data, pred, coords=coords, title=name)
        if save_detection:
            os.makedirs(DETECTION_OUT_ROOT, exist_ok=True)
            coords = np.concatenate([np.arange(0, len(coords))[:, None], coords], axis=1)
            coords = pd.DataFrame(coords, columns=["index", "axis-0", "axis-1", "axis-2"])
            fname = Path(path).stem
            detection_save_path = os.path.join(DETECTION_OUT_ROOT, f"{fname}.csv")
            coords.to_csv(detection_save_path, index=False)


# TODO update to support post-processing and showing annotations for the val data
def main():
    # check_val_image()
    check_new_images(view=False, save_detection=True)


if __name__ == "__main__":
    main()
