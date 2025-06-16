import os

import imageio.v3 as imageio
import pandas as pd

from flamingo_tools.segmentation.marker_detection import map_and_filter_detections


ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/croppings/Synapse_crop"
VGLUT_PATH = os.path.join(ROOT, "M_LR_000226_R_crop_1098-0926-0872_Vglut3.tif")
CTBP2_PATH = os.path.join(ROOT, "M_LR_000226_R_crop_1098-0926-0872_CTBP2.tif")
SEG_PATH = os.path.join(ROOT, "M_LR_000226_R_resized_crop_1098-0926-0872_IHC.tif")
DET_PATH = os.path.join(ROOT, "synapses/synapse_detection.tsv")


def check_data():
    import napari

    detections = pd.read_csv(DET_PATH, sep="\t")
    detections = detections[["z", "y", "x"]].values

    filtered_path = os.path.join(ROOT, "synapses/synapse_detection_filtered.tsv")
    filtered_detections = pd.read_csv(filtered_path, sep="\t")
    filtered_detections = filtered_detections[["z", "y", "x"]].values

    vglut = imageio.imread(VGLUT_PATH)
    ctbp2 = imageio.imread(CTBP2_PATH)
    ihcs = imageio.imread(SEG_PATH)

    v = napari.Viewer()
    v.add_image(vglut)
    v.add_image(ctbp2)
    v.add_labels(ihcs)
    v.add_points(detections)
    v.add_points(filtered_detections)
    napari.run()


def map_synapses():
    ihcs = imageio.imread(SEG_PATH)
    detections = pd.read_csv(DET_PATH, sep="\t")
    n_detections = len(detections)

    detections = map_and_filter_detections(ihcs, detections, max_distance=2.0, n_threads=8)
    print("Detections after mapping and fitering:", len(detections), "/", n_detections)

    out_path = os.path.join(ROOT, "synapses/synapse_detection_filtered.tsv")
    detections.to_csv(out_path, sep="\t", index=False)


def main():
    map_synapses()
    # check_data()


if __name__ == "__main__":
    main()
