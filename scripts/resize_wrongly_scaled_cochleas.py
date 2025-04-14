import argparse
import os

import multiprocessing as mp
from concurrent import futures

import nifty.tools as nt
from tqdm import tqdm

from elf.wrapper.resized_volume import ResizedVolume
from elf.io import open_file
from flamingo_tools.file_utils import read_tif


def main(input_path, output_folder, scale, input_key, interpolation_order):
    if input_path.endswith(".tif"):
        input_ = read_tif(input_path)
        input_chunks = (128,) * 3
    else:
        input_ = open_file(input_path, "r")[input_key]
        input_chunks = input_.chunks

    abs_path = os.path.abspath(input_path)
    basename = "".join(os.path.basename(abs_path).split(".")[:-1])
    output_path = os.path.join(output_folder, basename + "_resized.n5")

    shape = input_.shape
    ndim = len(shape)

    # Limit the number of cores for parallelization.
    n_threads = min(16, mp.cpu_count())

    shape = input_.shape
    new_shape = tuple(
        int(round(sh / scale)) for sh in shape
    )

    resized_volume = ResizedVolume(input_, new_shape, order=interpolation_order)

    output = open_file(output_path, mode="a")
    output_dataset = output.create_dataset(
        input_key, shape=new_shape, dtype=input_.dtype,
        chunks=input_chunks, compression="gzip"
    )
    blocking = nt.blocking([0] * ndim, new_shape, input_chunks)

    def copy_chunk(block_index):
        block = blocking.getBlock(block_index)
        volume_index = tuple(slice(begin, end) for (begin, end) in zip(block.begin, block.end))
        data = resized_volume[volume_index]
        output_dataset[volume_index] = data

    with futures.ThreadPoolExecutor(n_threads) as resize_pool:
        list(tqdm(
            resize_pool.map(copy_chunk, range(blocking.numberOfBlocks)),
            total=blocking.numberOfBlocks,
            desc=f"Resizing volume from shape {shape} to {new_shape}"
        ))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for resizing microscoopy data in n5 format.")

    parser.add_argument("input_file", type=str, help="Input file")
    parser.add_argument(
        "output_folder", type=str, help="Output folder. Default resized output is <basename>_resized.n5"
    )

    parser.add_argument("-s", "--scale", type=float, default=0.38, help="Scale of input. Re-scaled to 1.")
    parser.add_argument("-k", "--input_key", type=str, default="setup0/timepoint0/s0", help="Input key for n5 file.")
    parser.add_argument("-i", "--interpolation_order", type=float, default=3, help="Interpolation order.")

    args = parser.parse_args()

    main(args.input_file, args.output_folder, args.scale, args.input_key, args.interpolation_order)
