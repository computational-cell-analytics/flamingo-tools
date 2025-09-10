import multiprocessing as mp
from concurrent import futures
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import zarr

from elf.io import open_file
from elf.parallel.local_maxima import find_local_maxima
from flamingo_tools.segmentation.unet_prediction import prediction_impl
from tqdm import tqdm


def sgn_detection(
    input_path: str,
    input_key: str,
    output_folder: str,
    model_path: str,
    block_shape: Optional[Tuple[int, int, int]] = None,
    halo: Optional[Tuple[int, int, int]] = None,
    spot_radius: int = 4,
):
    """Run prediction for sgn detection.

    Args:
        input_path: Input path to image channel for SGN detection.
        input_key: Input key for resolution of image channel and mask channel.
        output_folder: Output folder for SGN segmentation.
        model_path: Path to model for SGN detection.
        block_shape: The block-shape for running the prediction.
        halo: The halo (= block overlap) to use for prediction.
        spot_radius: Radius in pixel to convert spot detection of SGNs into a volume.
    """
    if block_shape is None:
        block_shape = (24, 256, 256)
    if halo is None:
        halo = (12, 64, 64)

    # Skip existing prediction, which is saved in output_folder/predictions.zarr
    skip_prediction = False
    output_path = os.path.join(output_folder, "predictions.zarr")
    prediction_key = "prediction"
    if os.path.exists(output_path) and prediction_key in zarr.open(output_path, "r"):
        skip_prediction = True

    if not skip_prediction:
        prediction_impl(
            input_path, input_key, output_folder, model_path,
            scale=None, block_shape=block_shape, halo=halo,
            apply_postprocessing=False, output_channels=1,
        )

    detection_path = os.path.join(output_folder, "SGN_detection.tsv")
    detection_path = os.path.join(output_folder, "SGN_detection.tsv")
    if not os.path.exists(detection_path):
        input_ = zarr.open(output_path, "r")[prediction_key]
        detections = find_local_maxima(
            input_, block_shape=block_shape, min_distance=4, threshold_abs=0.5, verbose=True, n_threads=16,
        )

        print(detections.shape)

        shape = input_.shape
        chunks = (128, 128, 128)
        segmentation_path = os.path.join(output_folder, "segmentation.zarr")
        output = open_file(segmentation_path, mode="a")
        segmentation_key = "segmentation"
        output_dataset = output.create_dataset(
            segmentation_key, shape=shape, dtype=input_.dtype,
            chunks=chunks, compression="gzip"
        )

        def add_halo_segm(detection_index):
            """Create a segmentation volume around all detected spots.
            """
            coord = detections[detection_index]
            block_begin = [round(c) - spot_radius for c in coord]
            block_end = [round(c) + spot_radius for c in coord]
            volume_index = tuple(slice(beg, end) for beg, end in zip(block_begin, block_end))
            output_dataset[volume_index] = detection_index + 1

        # Limit the number of cores for parallelization.
        n_threads = min(16, mp.cpu_count())
        with futures.ThreadPoolExecutor(n_threads) as filter_pool:
            list(tqdm(filter_pool.map(add_halo_segm, range(len(detections))), total=len(detections)))

        # Save the result in mobie compatible format.
        detections = np.concatenate(
            [np.arange(1, len(detections) + 1)[:, None], detections[:, ::-1]], axis=1
        )
        detections = pd.DataFrame(detections, columns=["spot_id", "x", "y", "z"])
        detections.to_csv(detection_path, index=False, sep="\t")
