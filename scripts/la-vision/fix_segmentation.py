import os
from glob import glob

import napari
import numpy as np
import imageio.v3 as imageio
import vigra

from skimage.filters import gaussian
from skimage.segmentation import find_boundaries, watershed
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label


def _size_filter(segmentation, heightmap, min_size):
    ids, sizes = np.unique(segmentation, return_counts=True)
    discard_ids = ids[sizes < min_size]
    mask = segmentation > 0
    segmentation[np.isin(segmentation, discard_ids)] = 0
    return watershed(heightmap, markers=segmentation, mask=mask)


def postproc(image, segmentation, view=False):
    # First get rid of small objects.
    min_size = 250
    heightmap = vigra.filters.laplacianOfGaussian(image.astype("float32"), 3)

    segmentation = _size_filter(segmentation, heightmap, min_size)

    mask = ~find_boundaries(segmentation)
    dist = distance_transform_edt(mask, sampling=(2, 1, 1))
    dist[segmentation == 0] = 0
    dist = gaussian(dist, (0.6, 1.2, 1.2))
    maxima = peak_local_max(dist, min_distance=3, exclude_border=False)

    maxima_image = np.zeros(segmentation.shape, dtype="uint8")
    pos = tuple(maxima[:, i] for i in range(3))
    maxima_image[pos] = 1
    maxima_image = label(maxima_image)

    def maxima_ids(seg, im):
        ids = np.unique(im[seg])
        return ids[1:]

    seed_maxima_ids, keep_seg_ids, split_seg_ids = [], [], []
    props = regionprops(segmentation, maxima_image, extra_properties=[maxima_ids])
    for prop in props:
        this_maxima_ids = prop.maxima_ids
        if len(this_maxima_ids) == 1:
            keep_seg_ids.append(prop.label)
            continue
        seed_maxima_ids.extend(this_maxima_ids.tolist())
        split_seg_ids.append(prop.label)

    split_mask = np.isin(segmentation, split_seg_ids)
    # segmentation[split_mask] = 0

    new_seeds = maxima_image.copy()
    new_seeds[~np.isin(maxima_image, seed_maxima_ids)] = 0
    new_seg = watershed(heightmap, markers=new_seeds, mask=split_mask)

    segmentation[split_mask] = 0
    offset = segmentation.max()
    new_seg[new_seg != 0] += offset
    segmentation[split_mask] = new_seg[split_mask]
    segmentation = label(segmentation)
    segmentation = _size_filter(segmentation, heightmap, min_size)

    if view:
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(segmentation)
        # v.add_labels(new_seg)
        # v.add_image(heightmap)
        # v.add_image(dist)
        # v.add_points(maxima)
        # v.add_labels(split_mask)
        napari.run()

    return segmentation


def postprocess_volume(im_path, seg_path, out_root):
    image = imageio.imread(im_path)
    segmentation = imageio.imread(seg_path)
    segmentation = postproc(image, segmentation, view=True)

    os.makedirs(out_root, exist_ok=True)
    fname = os.path.basename(im_path)
    imageio.imwrite(os.path.join(out_root, fname), segmentation, compression="zlib")


def postprocess_volume_scalable(im_path, seg_path, out_root):
    from flamingo_tools.segmentation.postprocessing import split_nonconvex_objects, compute_table_on_the_fly

    image = imageio.imread(im_path)
    segmentation = imageio.imread(seg_path)

    # TODO aniso resolution
    resolution = 0.38
    table = compute_table_on_the_fly(segmentation, resolution)

    out = np.zeros_like(segmentation)
    id_mapping = split_nonconvex_objects(segmentation, out, table, n_threads=1, resolution=resolution, min_size=250)
    n_prev = len(id_mapping)
    n_after = sum([len(v) for v in id_mapping.values()])
    print("Before splitting:", n_prev)
    print("After splitting:", n_after)

    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(segmentation, visible=False)
    v.add_labels(out)
    napari.run()


def main():
    im_paths = sorted(glob("la-vision-sgn-new/images/*.tif"))
    seg_paths = sorted(glob("la-vision-sgn-new/segmentation/*.tif"))
    out_root = "la-vision-sgn-new/segmentation-postprocessed"
    for im_path, seg_path in zip(im_paths, seg_paths):
        # postprocess_volume(im_path, seg_path, out_root)
        postprocess_volume_scalable(im_path, seg_path, out_root)
        break


if __name__ == "__main__":
    main()
