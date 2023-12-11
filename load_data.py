import numpy as np
import z5py


def load_data(file_path, key, bounding_box):
    """Load data out of an n5 file from a region (defauned by bounding_box) into memory.
    """

    # The z5py library implements the same interface as h5py (for HDF5 files).
    # For details check out:
    # -
    # -

    # Open the file handle.
    with z5py.File(file_path, "r") as n5_file:

        # Get the hanlde to the dataset (which contains the actual data).
        dataset = n5_file[key]
        # Print some info about the dataset.
        print("The dataset @", file_path, ":", key)
        print("has the shape:", dataset.shape)
        print("and chunks:", dataset.chunks)

        # Load the region corresponding to bounding_box into memory
        sub_volume = dataset[:]

    return sub_volume


def check_data(sub_volume):
    """Check the loaded data visually with napari.

    Note that napari is by default not installed in the python environment.
    It can easily be installed via mamba. Check out https://napari.org/stable/ for details.
    """
    import napari
    v = napari.Viewer()
    v.add_image(sub_volume)
    napari.run()


def main():
    file_path = "./converted/converted.n5"
    # The key names are specific for how the BigDataViewer stores
    # Setups, timepoints and scale levels.
    # To load a different setup etc. change the corresponding index.
    key = "setup0/timepoint0/s1"

    # Set the bounding box for the sub-volume that will be loaded. For example:
    # Load the first 100 slices.
    bounding_box = np.s_[0:100, :, :]
    # Or load the complete volume.
    bounding_box = np.s_[:]

    sub_volume = load_data(file_path, key, bounding_box)
    check_data(sub_volume)


if __name__ == "__main__":
    main()
