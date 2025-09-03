import os
from glob import glob

from tifffile import imread, imwrite
from csbdeep.utils import normalize
from stardist.models import StarDist3D

model = StarDist3D.from_pretrained("3D_demo")


def segment_with_stardist(ff, out):
    axis_norm = (0, 1, 2)  # normalize channels independently
    x = imread(ff)
    img = normalize(x[0], 1, 99.8, axis=axis_norm)
    labels, details = model.predict_instances(img)
    imwrite(out, labels)


def main():
    files = glob("predictions/sgn-new/*.tif")
    out_folder = "./predictions/stardist"
    os.makedirs(out_folder, exist_ok=True)
    for ff in files:
        out = imread(ff)
        segment_with_stardist(ff, out)


if __name__ == "__main__":
    main()
