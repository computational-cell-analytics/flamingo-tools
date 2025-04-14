"""Prediction using distance U-Net.
Parallelization using multiple GPUs is currently only possible
by calling functions located in segmentation/unet_prediction.py directly.
Functions for the parallelization end with '_slurm' and divide the process into preprocessing,
prediction, and segmentation.
"""
import argparse

import torch
import z5py


def main():
    from flamingo_tools.segmentation import run_unet_prediction

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-k", "--input_key", default=None)
    parser.add_argument("-s", "--scale", default=None, type=float, help="Downscale the image by the given factor.")
    parser.add_argument("-b", "--block_shape", default=None, type=int, nargs=3)

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
        if args.block_shape is None:
            block_shape = (64, 256, 256) if have_cuda else (64, 64, 64)
        else:
            block_shape = tuple(args.block_shape)
        halo = (16, 64, 64) if have_cuda else (8, 32, 32)
    else:
        if args.block_shape is None:
            chunks = z5py.File(args.input, "r")[args.input_key].chunks
            block_shape = tuple([2 * ch for ch in chunks]) if have_cuda else tuple(chunks)
        else:
            block_shape = tuple(args.block_shape)
        halo = (16, 64, 64) if have_cuda else (8, 32, 32)

    run_unet_prediction(
        args.input, args.input_key, args.output_folder, args.model,
        scale=scale, min_size=min_size,
        block_shape=block_shape, halo=halo,
    )


if __name__ == "__main__":
    main()
