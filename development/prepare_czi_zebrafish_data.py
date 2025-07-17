def get_czi_zebrafish_data():
    # NOTE: Let's try for one link first, we can generalize it later.
    url = "https://public.czbiohub.org/royerlab/ultrack/zebrafish_embryo.ome.zarr"

    from ome_zarr.reader import Reader
    from ome_zarr.io import parse_url

    reader = Reader(parse_url(url))  # Prepare a reader.
    nodes = list(reader())  # Might include multiple stuff
    image_node = nodes[0]  # First node is expecte to be image pixel data.

    dask_data = image_node.data  # Get the daskified data.

    # HACK: Try it for one dask array with lowest resolution.
    curr_data = dask_data[-1]

    # Load using napari
    import napari
    viewer = napari.view_image(curr_data, channel_axis=0)
    napari.run()


def main():
    get_czi_zebrafish_data()


if __name__ == "__main__":
    main()
