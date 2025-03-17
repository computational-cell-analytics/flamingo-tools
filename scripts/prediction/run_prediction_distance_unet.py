import argparse
import sys

import torch
import z5py

sys.path.append("../..")


def main():
    from flamingo_tools.segmentation import run_unet_prediction

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-k", "--input_key", default=None)
    parser.add_argument("-s", "--scale", default=None, type=float, help="Downscale the image by the given factor.")
    parser.add_argument("-n", "--number_gpu", default=1, type=int, help="Number of GPUs to use in parallel.")

    args = parser.parse_args()

    scale = args.scale
    if scale is None:
        min_size = 1000
    elif scale > 1:
        min_size = 250
    elif scale < 1:
        min_size = 1000

    have_cuda = torch.cuda.is_available()
    if args.input_key is None:
        block_shape = (64, 256, 256) if have_cuda else (64, 64, 64)
        halo = (16, 64, 64) if have_cuda else (8, 32, 32)
    else:
        chunks = z5py.File(args.input, "r")[args.input_key].chunks
        block_shape = tuple([2 * ch for ch in chunks]) if have_cuda else tuple(chunks)
        halo = (16, 64, 64) if have_cuda else (8, 32, 32)

    prediction_instances = args.number_gpu if have_cuda else 1

    run_unet_prediction(
        args.input, args.input_key, args.output_folder, args.model,
        scale=scale, min_size=min_size,
        block_shape=block_shape, halo=halo,
        prediction_instances=prediction_instances,
    )


if __name__ == "__main__":
    main()
