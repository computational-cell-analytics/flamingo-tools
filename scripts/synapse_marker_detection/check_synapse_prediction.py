import h5py
import napari
import zarr

from torch_em.util import load_model
from torch_em.util.prediction import predict_with_halo
from train_synapse_detection import get_paths


def run_prediction(val_image):
    model = load_model("./checkpoints/synapse_detection_v1")
    block_shape = (32, 384, 384)
    halo = (8, 64, 64)
    pred = predict_with_halo(val_image, model, [0], block_shape, halo)
    return pred.squeeze()


def main():
    val_paths, _ = get_paths("val")
    val_image = zarr.open(val_paths[0])["raw"][:]

    # pred = run_prediction(val_image)
    # with h5py.File("pred.h5", "a") as f:
    #     f.create_dataset("pred", data=pred, compression="gzip")

    with h5py.File("pred.h5", "r") as f:
        pred = f["pred"][:]

    v = napari.Viewer()
    v.add_image(val_image)
    v.add_image(pred)
    napari.run()


if __name__ == "__main__":
    main()
