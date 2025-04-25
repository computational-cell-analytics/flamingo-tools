import argparse
import os

import pandas as pd
import numpy as np
import zarr

from elf.parallel.local_maxima import find_local_maxima
from flamingo_tools.segmentation.unet_prediction import prediction_impl


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-k", "--input_key", default=None)
    args = parser.parse_args()

    block_shape = (64, 256, 256)
    halo = (16, 64, 64)

    # Skip existing prediction, which is saved in output_folder/predictions.zarr
    skip_prediction = False
    output_path = os.path.join(args.output_folder, "predictions.zarr")
    prediction_key = "prediction"
    if os.path.exists(output_path) and prediction_key in zarr.open(output_path, "r"):
        skip_prediction = True

    if not skip_prediction:
        prediction_impl(
            args.input, args.input_key, args.output_folder, args.model,
            scale=None, block_shape=block_shape, halo=halo,
            apply_postprocessing=False, output_channels=1,
        )

    detection_path = os.path.join(args.output_folder, "synapse_detection.tsv")
    if not os.path.exists(detection_path):
        input_ = zarr.open(output_path, "r")[prediction_key]
        detections = find_local_maxima(
            input_, block_shape=block_shape, min_distance=2, threshold_abs=0.5, verbose=True, n_threads=16,
        )
        # Save the result in mobie compatible format.
        detections = np.concatenate(
            [np.arange(1, len(detections) + 1)[:, None], detections[:, ::-1]], axis=1
        )
        detections = pd.DataFrame(detections, columns=["spot_id", "x", "y", "z"])
        detections.to_csv(detection_path, index=False, sep="\t")


if __name__ == "__main__":
    main()
