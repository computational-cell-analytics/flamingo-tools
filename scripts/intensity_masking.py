import argparse

import imageio.v3 as imageio
import numpy as np
from scipy.ndimage import binary_dilation, binary_closing, distance_transform_edt


def intensity_masking(image_path, seg_path, out_path, modulation_strength=10, dilation=2, view=False):
    seg = imageio.imread(seg_path)
    mask = binary_dilation(seg != 0, iterations=2)
    mask = binary_closing(mask, iterations=4)

    image = imageio.imread(image_path)
    lo, hi = np.percentile(image, 2), np.percentile(image, 98)
    print(lo, hi)
    image_modulated = np.clip(image, lo, hi).astype("float32")
    image_modulated -= lo
    image_modulated /= image_modulated.max()

    modulation_mask = distance_transform_edt(~mask)
    modulation_mask /= modulation_mask.max()
    modulation_mask = 1 - modulation_mask
    modulation_mask[mask] = 1
    modulation_mask = np.pow(modulation_mask, 3)
    modulation_mask *= modulation_strength
    image_modulated *= modulation_mask

    if view:
        import napari
        v = napari.Viewer()
        v.add_image(modulation_mask)
        v.add_image(image, visible=False)
        v.add_image(image_modulated)
        v.add_labels(mask, visible=False)
        napari.run()
        return
    imageio.imwrite(out_path, image_modulated, compression="zlib")


# image_path = "M_LR_000227_R/scale3/PV.tif"
# seg_path = "M_LR_000227_R/scale3/SGN_v2.tif"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("seg_path")
    parser.add_argument("out_path")
    parser.add_argument("--view", "-v", action="store_true")
    parser.add_argument("--dilation", type=int, default=2)
    args = parser.parse_args()
    intensity_masking(args.image_path, args.seg_path, args.out_path, view=args.view, dilation=args.dilation)
