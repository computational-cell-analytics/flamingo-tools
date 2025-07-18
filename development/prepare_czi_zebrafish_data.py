import os
import subprocess
from typing import Tuple

import pandas as pd

from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

import dask.array as da


def _get_dasky_data(url):
    reader = Reader(parse_url(url))  # Prepare a reader.
    nodes = list(reader())  # Might include multiple stuff
    image_node = nodes[0]  # First node is expected to be image pixel data.

    dask_data = image_node.data  # Get the daskified data.

    return dask_data


def get_zebrahub_data(timepoint: int = 100, view: bool = False) -> Tuple[da.Array, pd.DataFrame]:
    """Gets the ZebraHub data from https://doi.org/10.1016/j.cell.2024.09.047.

    Args:
        timepoint: The timepoint where the 3d imaging data will be returned from.
        view: Whether to view the dask array via napari.

    Returns:
        The daskified chunky array.
        And the tracking annotations.
    """
    # NOTE: There's more single objective samples for zebrafish available with tracking annotations
    # https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/
    url = "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001.ome.zarr"

    # Let's get the image data.
    dask_data = _get_dasky_data(url)

    # Get the lowest resolution (see below on how to access other resolutions)
    curr_data = dask_data[-1]

    # And strip out the channel dimension (see below for more details)
    curr_data = curr_data[timepoint, 0]

    # We have tracking annotations here. Let's check them out.
    tracks_fpath = "ZSNS001_tracks.csv"
    if not os.path.exists(tracks_fpath):
        subprocess.run(
            ["wget", "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001_tracks.csv"]
        )

    # Load the tracking annotation file.
    tracks = pd.read_csv("ZSNS001_tracks.csv")  # I think this is on original resolution (?)

    # HACK: Filtering ids based on one time-frame (the most plausible setup we might be opting for)
    curr_tracks = tracks.loc[tracks["t"] == timepoint]

    if view:
        import napari
        napari.view_image(curr_data)
        napari.run()

    return curr_data, curr_tracks


def get_czi_zebrafish_data(
    neuromast: bool = True, view: bool = False
) -> da.Array:
    """Gets the CZI ZebraFish light-sheet microscopy data.
    NOTE: Currently, we support only the raw data.

    Args:
        view: Whether to view the dask array via napari.

    Returns:
        The daskified chunky array.
    """
    # NOTE: Let's try for one link first, we can generalize it later.

    if neuromast:
        # Link for nuclear and membrane labeled zebrafish neuromast.
        url = "https://public.czbiohub.org/royerlab/ultrack/zebrafish_neuromast.ome.zarr"
    else:
        # Link for dense nuclear labeled zebrafish embryo.
        # NOTE: This data does not have tracking annotations!
        url = "https://public.czbiohub.org/royerlab/ultrack/zebrafish_embryo.ome.zarr"

    # Let's get the image data
    dask_data = _get_dasky_data(url)

    # HACK: Try it for one dask array with lowest resolution (there exists four resolutions in this data).
    # TODO: Control res below, the highest res starts at the first index, lowest at the last index.
    curr_data = dask_data[-1]

    # We don't care about the over-time information. Let's get the 3d info for now!
    # I am removing the channel dimension here (OG dimension style: (T, C, Z, Y, X))
    curr_data = curr_data[:, 0]  # TODO: Parse values in the time or z-dimension to access limited slices?

    # NOTE: The following line of code brings the entire dask array in memory.
    # curr_data = curr_data.compute()

    if view:
        import napari
        napari.view_image(curr_data)
        napari.run()

    return curr_data


def main():
    # image = get_czi_zebrafish_data(neuromast=True, view=False)

    # Suggested timepoints I like in the developmental cycle:
    # 740/760: kind of at the end of cycle.
    # 650: it's a nice development stage which visually surfaces a lot of nucleus.

    image, tracks = get_zebrahub_data(timepoint=650, view=False)
    print(image.shape)


if __name__ == "__main__":
    main()
