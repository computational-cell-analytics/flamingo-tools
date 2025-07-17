from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

import dask.array as da


def get_czi_zebrafish_data(view: bool = False) -> da.Array:
    """Gets the CZI ZebraFish light-sheet microscopy data.
    NOTE: Currently, we support only the raw data.

    Args:
        view: Whether to view the dask array via napari.

    Returns:
        The daskified chunky array.
    """
    # NOTE: Let's try for one link first, we can generalize it later.
    url = "https://public.czbiohub.org/royerlab/ultrack/zebrafish_embryo.ome.zarr"

    reader = Reader(parse_url(url))  # Prepare a reader.
    nodes = list(reader())  # Might include multiple stuff
    image_node = nodes[0]  # First node is expecte to be image pixel data.

    dask_data = image_node.data  # Get the daskified data.

    # HACK: Try it for one dask array with lowest resolution (there exists four resolutions in this data).
    curr_data = dask_data[-1]  # TODO: Control dimensions from here, the highest res starts at the first index.

    # We don't care about the over-time information. Let's get the 3d info for now!
    # I am removing the channel dimension here (OG dimension style: (T, C, Z, Y, X))
    curr_data = curr_data[:, 0]  # TODO: Parse values in the time or z-dimension to parse limited slices?

    # NOTE: The following line of code brings the entire dask array in memory.
    # curr_data = curr_data.compute()

    if view:
        import napari
        napari.view_image(curr_data)
        napari.run()

    return curr_data


def main():
    image = get_czi_zebrafish_data(view=False)
    print(image.shape)


if __name__ == "__main__":
    main()
