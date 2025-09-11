import os
import imageio.v3 as imageio
from pathlib import Path

import pandas as pd
import torch
from skimage.feature import peak_local_max
from torch_em.util.prediction import predict_with_halo

ims = [
    "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/SGN/sgn-detection/images/LaVision-M04_crop_2580-2266-0533_PV.tif",
    "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/SGN/sgn-detection/empty_images/LaVision-M04_crop_0400-2500-0840_PV_empty.tif"
]

model_path = "checkpoints/sgn-detection.pt"
model = torch.load(model_path, weights_only=False)

block_shape = [24, 256, 256]
halo = (8, 64, 64)

out = "./detections-v1"
os.makedirs(out, exist_ok=True)
for im in ims:
    data = imageio.imread(im)
    pred = predict_with_halo(data, model, [0], block_shape, halo).squeeze()

    coords = peak_local_max(pred, min_distance=4, threshold_abs=0.5)

    # coords = np.concatenate([np.arange(0, len(coords))[:, None], coords], axis=1)
    coords = pd.DataFrame(coords, columns=["axis-0", "axis-1", "axis-2"])

    name = Path(im).stem
    imageio.imwrite(os.path.join(out, f"{name}.tif"), pred)
    coords.to_csv(os.path.join(out, f"{name}.csv"), index=False)
