import argparse
import multiprocessing as mp
import os
from concurrent import futures

import imageio.v3 as imageio
import elf.parallel as parallel
import numpy as np
import vigra
import torch

from elf.wrapper import ThresholdWrapper, SimpleTransformationWrapper
from elf.wrapper.resized_volume import ResizedVolume
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


def prediction_impl(input_path, input_key, output_path, model_path, scale):
    model = load_model(model_path)

    if input_key is None:
        input_ = imageio.imread(input_path)
    else:
        input_ = open_file(input_path, "r")[input_key]

    if scale is None:
        original_shape = None
    else:
        original_shape = input_.shape
        new_shape = tuple(
            int(round(sh / scale)) for sh in original_shape
        )
        print("The input is processed downsampled by a factor of scale", scale)
        print("Corresponding to shape", new_shape, "instead of", original_shape)
        input_ = ResizedVolume(input_, shape=new_shape, order=3)

    if torch.cuda.is_available():
        print("Predict with GPU")
        gpu_ids = [0]
        block_shape = (96, 256, 256)
        halo = (32, 64, 64)
    else:
        print("Predict with CPU")
        gpu_ids = ["cpu"]
        block_shape = (64, 96, 96)
        halo = (16, 32, 32)

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

    return original_shape


def smoothing(x, sigma=2.0):
    try:
        smoothed = vigra.filters.gaussianSmoothing(x, sigma)
    except RuntimeError:
        smoothed = x
    return smoothed


def segmentation_impl(input_path, output_folder, min_size=2000, original_shape=None):
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
    if original_shape is None:
        seg_path = os.path.join(output_folder, "segmentation.zarr")
    else:
        seg_path = os.path.join(output_folder, "seg_downscaled.zarr")

    seg_file = open_file(seg_path, "a")
    seg = seg_file.create_dataset(
        "segmentation", shape=seeds.shape, chunks=block_shape, compression="gzip", dtype="uint64"
    )

    hmap = SelectChannel(input_, 2)
    halo = (2, 8, 8)
    # Limit the number of cores for seeded watershed, which is otherwise quite memory hungry.
    n_threads_ws = min(8, mp.cpu_count())
    parallel.seeded_watershed(
        hmap, seeds, out=seg, block_shape=block_shape, halo=halo, mask=mask, verbose=True,
        n_threads=n_threads_ws,
    )

    if min_size > 0:
        parallel.size_filter(seg, seg, min_size=min_size, block_shape=block_shape, mask=mask, verbose=True)

    if original_shape is not None:
        out_path = os.path.join(output_folder, "segmentation.zarr")

        # This logic should be refactored.
        output_seg = ResizedVolume(seg, shape=original_shape, order=0)
        with open_file(out_path, "a") as f:
            out_seg_volume = f.create_dataset(
                "segmentation", shape=original_shape, compression="gzip", dtype="uint64", chunks=block_shape,
            )
            n_threads = mp.cpu_count()
            blocking = parallel.common.get_blocking(output_seg, block_shape, roi=None, n_threads=n_threads)

            def write_block(block_id):
                block = blocking.getBlock(block_id)
                bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
                out_seg_volume[bb] = output_seg[bb]

            with futures.ThreadPoolExecutor(n_threads) as tp:
                tp.map(write_block, range(blocking.numberOfBlocks))


def run_prediction(input_path, input_key, output_folder, model_path, scale):
    os.makedirs(output_folder, exist_ok=True)

    pmap_out = os.path.join(output_folder, "predictions.zarr")
    original_shape = prediction_impl(input_path, input_key, pmap_out, model_path, scale)

    segmentation_impl(pmap_out, output_folder, original_shape=original_shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-k", "--input_key", default=None)
    parser.add_argument("-s", "--scale", default=None, type=float, help="Downscale the image by the given factor.")

    args = parser.parse_args()
    run_prediction(args.input, args.input_key, args.output_folder, args.model, scale=args.scale)


if __name__ == "__main__":
    main()
