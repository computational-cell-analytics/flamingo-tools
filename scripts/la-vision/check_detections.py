import napari
import zarr


resolution = [3.0, 1.887779, 1.887779]
positions = [
    [2002.95539395823, 1899.9032205156411, 264.7747008147759]
]


def _load_from_mobie(bb):
    path = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/LaVision-M04/images/ome-zarr/PV.ome.zarr"
    f = zarr.open(path, mode="r")
    data = f["s0"][bb]
    print(bb)

    path = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/LaVision-M04/images/ome-zarr/SGN_detect-v1.ome.zarr"
    f = zarr.open(path, mode="r")
    seg = f["s0"][bb]

    return data, seg


def _load_prediction(bb):
    path = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/LaVision-M04/SGN_detect-v1/predictions.zarr"
    f = zarr.open(path, mode="r")
    data = f["prediction"][bb]
    return data


def _load_prediction_debug():
    path = "./debug-pred/pred-v5.h5"
    with zarr.open(path, "r") as f:
        pred = f["pred"][:]
    return pred


def check_detection(position, halo=[32, 384, 384]):

    bb = tuple(
        slice(int(pos / re) - ha, int(pos / re) + ha) for pos, re, ha in zip(position[::-1], resolution, halo)
    )

    pv, detections_mobie = _load_from_mobie(bb)
    # pred = _load_prediction(bb)
    pred = _load_prediction_debug()

    v = napari.Viewer()
    v.add_image(pv)
    v.add_image(pred)
    v.add_labels(detections_mobie)
    napari.run()


def main():
    position = positions[0]
    check_detection(position)


if __name__ == "__main__":
    main()
