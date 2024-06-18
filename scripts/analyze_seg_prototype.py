import imageio.v3 as imageio
import napari
import z5py


def main():
    seg_path = "/home/pape/Work/data/moser/lightsheet/new-vol/seg.tif"
    raw_path = "/home/pape/Work/data/moser/lightsheet/new-vol/vol.n5"

    print("Loading segmentation ...")
    seg = imageio.imread(seg_path)
    print(seg.shape)

    print("Loading raw data ...")
    with z5py.File(raw_path, "r") as f:
        raw = f["setup0/s0"]
        raw.n_threads = 8
        raw = raw[:]
    print(raw.shape)

    print("Start viewer ...")
    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(seg)
    napari.run()


if __name__ == "__main__":
    main()
