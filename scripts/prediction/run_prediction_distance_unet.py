import argparse
import os

import imageio.v3 as imageio
import elf.parallel as parallel
import numpy as np
import vigra
import torch

from elf.wrapper import ThresholdWrapper, SimpleTransformationWrapper
from elf.io import open_file
from torch_em.util import load_model
from torch_em.util.prediction import predict_with_halo


class SelectChannel(SimpleTransformationWrapper):
    def __init__(self, volume, channel):
        self.channel = channel
        super().__init__(volume, lambda x: x[self.channel], with_channels=True)

    @property
    def shape(self):
        return self._volume.shape[1:]

    @property
    def chunks(self):
        return self._volume.chunks[1:]

    @property
    def ndim(self):
        return self._volume.ndim - 1


def prediction_impl(input_path, input_key, output_path, model_path):
    model = load_model(model_path)

    if input_key is None:
        input_ = imageio.imread(input_path)
    else:
        input_ = open_file(input_path, "r")[input_key]

    if torch.cuda.is_available():
        gpu_ids = ["cpu"]
        block_shape = (64, 96, 96)
        halo = (16, 32, 32)
    else:
        gpu_ids = [0]
        block_shape = (96, 256, 256)
        halo = (32, 64, 64)

    with open_file(output_path, "a") as f:
        output = f.require_dataset(
            "prediction",
            shape=(3,) + input_.shape,
            chunks=(1,) + block_shape,
            compression="gzip",
        )

        # TODO do the smoothing as post-processing here, so that we can remove it from
        # the segmentation in order to make that more robust
        predict_with_halo(
            input_, model,
            gpu_ids=gpu_ids, block_shape=block_shape, halo=halo,
            output=output,
        )


def smoothing(x, sigma=2.0):
    try:
        smoothed = vigra.filters.gaussianSmoothing(x, sigma)
    except RuntimeError:
        smoothed = x
    return smoothed


def segmentation_impl(input_path, output_folder, min_size=2000):
    input_ = open_file(input_path, "r")["prediction"]

    # The smoothed center distances as input for computing the seeds.
    center_distances = SimpleTransformationWrapper(  # Wrapper to apply smoothing per block.
        # Wrapper to select the channel corresponding to the distance predictions.
        SelectChannel(input_, 1), smoothing
    )

    block_shape = center_distances.chunks

    # Compute the seeds based on smoothed center distances < 0.5.
    seed_path = os.path.join(output_folder, "seeds.zarr")
    seed_file = open_file(os.path.join(seed_path), "a")
    seeds = seed_file.require_dataset(
        "seeds", shape=center_distances.shape, chunks=block_shape, compression="gzip", dtype="uint64"
    )

    mask = ThresholdWrapper(SelectChannel(input_, 0), threshold=0.5)

    parallel.label(
        data=ThresholdWrapper(center_distances, threshold=0.4, operator=np.less),
        out=seeds, block_shape=block_shape, mask=mask, verbose=True,
    )

    # Run the watershed.
    seg_path = os.path.join(output_folder, "segmentation.zarr")
    seg_file = open_file(seg_path, "a")
    seg = seg_file.create_dataset(
        "segmentation", shape=seeds.shape, chunks=block_shape, compression="gzip", dtype="uint64"
    )

    hmap = SelectChannel(input_, 2)
    halo = (2, 8, 8)
    parallel.seeded_watershed(
        hmap, seeds, out=seg, block_shape=block_shape, halo=halo, mask=mask, verbose=True,
    )

    if min_size > 0:
        parallel.size_filter(seg, seg, min_size=min_size, block_shape=block_shape, mask=mask, verbose=True)


def run_prediction(input_path, input_key, output_folder, model_path):
    os.makedirs(output_folder, exist_ok=True)

    pmap_out = os.path.join(output_folder, "predictions.zarr")
    prediction_impl(input_path, input_key, pmap_out, model_path)

    segmentation_impl(pmap_out, output_folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-k", "--input_key", default=None)

    args = parser.parse_args()
    run_prediction(args.input, args.input_key, args.output_folder, args.model)


if __name__ == "__main__":
    main()
