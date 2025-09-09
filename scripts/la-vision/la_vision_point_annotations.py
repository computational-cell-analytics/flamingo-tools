import os
from glob import glob

import imageio.v3 as imageio
import napari
import numpy as np
from skimage.measure import regionprops


def main():
    image_files = sorted(glob("la-vision-sgn-new/images/*.tif"))
    label_files = sorted(glob("la-vision-sgn-new/segmentation-postprocessed/*.tif"))

    for imf, lf in zip(image_files, label_files):
        im = imageio.imread(imf)
        labels = imageio.imread(lf)

        props = regionprops(labels)
        centers = np.array([prop.centroid for prop in props])

        name = os.path.basename(imf)
        print(name)

        v = napari.Viewer()
        v.add_image(im)
        v.add_labels(labels)
        v.add_points(centers, size=5, out_of_slice_display=True)
        v.title = name
        napari.run()


if __name__ == "__main__":
    main()
