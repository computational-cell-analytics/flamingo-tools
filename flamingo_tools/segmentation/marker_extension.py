from threadpoolctl import threadpool_limits

import multiprocessing
from concurrent import futures
from threading import Lock
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from tqdm import tqdm

from elf.parallel.common import get_blocking


def distance_based_marker_extension(
    markers: np.ndarray,
    output: ArrayLike,
    extension_distance: float,
    sampling: Union[float, Tuple[float, ...]],
    block_shape: Tuple[int, ...],
    n_threads: Optional[int] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
):
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(output, block_shape, roi, n_threads)

    lock = Lock()

    # determine the correct halo in pixels based on the sampling and the extension distance.
    halo = [round(extension_distance / s) + 2 for s in sampling]

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def extend_block(block_id):
        block = blocking.getBlockWithHalo(block_id, halo)
        outer_block = block.outerBlock
        inner_block = block.innerBlock

        # TODO get the indices and coordinates of the markers in the INNER block
        # markers_in_block_ids = [int(i) for i in np.unique(inner_block)[1:]]
        mask = (
            (inner_block.begin[0] <= markers[:, 0]) & (markers[:, 0] <= inner_block.end[0]) &
            (inner_block.begin[1] <= markers[:, 1]) & (markers[:, 1] <= inner_block.end[1]) &
            (inner_block.begin[2] <= markers[:, 2]) & (markers[:, 2] <= inner_block.end[2])
        )
        markers_in_block_ids = np.where(mask)[0]
        markers_in_block_coords = markers[markers_in_block_ids]

        # TODO offset the marker coordinates with respect to the OUTER block
        markers_in_block_coords = [coord - outer_block.begin for coord in markers_in_block_coords]
        markers_in_block_coords = [[round(c) for c in coord] for coord in markers_in_block_coords]
        markers_in_block_coords = np.array(markers_in_block_coords, dtype=int)
        z, y, x = markers_in_block_coords.T

        # Shift index by one so that zero is reserved for background id
        markers_in_block_ids += 1

        # Create the seed volume.
        outer_block_shape = tuple(end - begin for begin, end in zip(outer_block.begin, outer_block.end))
        seeds = np.zeros(outer_block_shape, dtype="uint32")
        seeds[z, y, x] = markers_in_block_ids

        # Compute the distance map.
        distance = distance_transform_edt(seeds == 0, sampling=sampling)

        # And extend the seeds
        mask = distance < extension_distance
        segmentation = watershed(distance.max() - distance, markers=seeds, mask=mask)

        # Write the segmentation. Note: we need to lock here because we write outside of our inner block
        bb = tuple(slice(begin, end) for begin, end in zip(outer_block.begin, outer_block.end))
        with lock:
            this_output = output[bb]
            this_output[mask] = segmentation[mask]
            output[bb] = this_output

    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(extend_block, range(n_blocks)), total=n_blocks, desc="Marker extension", disable=not verbose
        ))
