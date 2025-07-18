from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

import dask.array as da


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

    # First, let's get the image data
    reader = Reader(parse_url(url))  # Prepare a reader.
    nodes = list(reader())  # Might include multiple stuff
    image_node = nodes[0]  # First node is expecte to be image pixel data.

    dask_data = image_node.data  # Get the daskified data.

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
    image = get_czi_zebrafish_data(neuromast=True, view=False)
    print(image.shape)


if __name__ == "__main__":
    main()
