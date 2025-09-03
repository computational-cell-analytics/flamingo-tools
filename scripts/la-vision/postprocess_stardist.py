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
    direc = "/mnt/vast-nhr/projects/nim00007/data/moser/predictions/sgn-new"
    out_folder = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/stardist"

    files = [entry.path for entry in os.scandir(direc) if ".tif" in entry.name]
    files.sort()
    file_names = [entry.name.split(".tif")[0] for entry in os.scandir(direc) if ".tif" in entry.name]
    file_names.sort()

    os.makedirs(out_folder, exist_ok=True)
    for f_path, f_name in zip(files, file_names):
        out = os.path.join(out_folder, f"{f_name}_seg.tif")
        segment_with_stardist(f_path, out)


if __name__ == "__main__":
    main()
