import os
from functools import partial

import numpy as np
import torch
import zarr
from torch_em.transform.raw import standardize
from torch_em.util.prediction import predict_with_halo


resolution = [3.0, 1.887779, 1.887779]
positions = [
    [2002.95539395823, 1899.9032205156411, 264.7747008147759]
]


def _load_from_mobie(bb):
    path = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/LaVision-M04/images/ome-zarr/PV.ome.zarr"
    f = zarr.open(path, mode="r")
    data = f["s0"][bb]
    return data


def run_prediction(position, halo=[32, 384, 384]):
    bb = tuple(
        slice(int(pos / re) - ha, int(pos / re) + ha) for pos, re, ha in zip(position[::-1], resolution, halo)
    )
    pv = _load_from_mobie(bb)
    mean, std = np.mean(pv), np.std(pv)
    print(mean, std)
    preproc = partial(standardize, mean=mean, std=std)

    block_shape = (24, 256, 256)
    halo = (8, 64, 64)

    model_path = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/SGN/sgn-detection-v1.pt"
    model = torch.load(model_path, weights_only=False)

    def postproc(x):
        x = np.clip(x, 0, 1)
        max_ = np.percentile(x, 99)
        x = x / max_
        return x

    pred = predict_with_halo(pv, model, [0], block_shape, halo, preprocess=preproc, postprocess=postproc).squeeze()

    pred_name = "pred-v5"
    out_folder = "./debug-pred"
    os.makedirs(out_folder, exist_ok=True)

    out_path = os.path.join(out_folder, f"{pred_name}.h5")
    with zarr.open(out_path, "w") as f:
        f.create_dataset("pred", data=pred)


def main():
    position = positions[0]
    run_prediction(position)


if __name__ == "__main__":
    main()
