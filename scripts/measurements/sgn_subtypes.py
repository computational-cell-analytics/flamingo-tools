import json
import os

import pandas as pd
from skimage.filters import threshold_otsu
from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target

# Map from cochlea names to channels
COCHLEAE_FOR_SUBTYPES = {
    "M_LR_000099_L": ["PV", "Calb1", "Lypd1"],
    "M_LR_000214_L": ["PV", "CR", "Calb1"],
    "M_AMD_N62_L": ["PV", "CR", "Calb1"],
    "M_AMD_Runx1_L": ["PV", "Lypd1", "Calb1"],
    # This one still has to be stitched:
    # "M_LR_000184_R": {"PV", "Prph"},
    # We don't have PV here, so we exclude these two for now.
    # "M_AMD_00N180_L": {"CR", "Ntng1", "Lypd1"},
    # "M_AMD_00N180_R": {"CR", "Ntng1", "CTBP2"},
}

# Map from channels to subtypes.
# Comment Aleyna:
# The signal will be a gradient between different subtypes:
# For example CR is expressed more, is brigther,
# in type 1a SGNs but exist in type Ib SGNs and to a lesser extent in type 1c.
# Same is also true for other markers so we will need to set a threshold for each.
# Luckily the signal seems less variable compared to GFP.
CHANNEL_TO_TYPE = {
    "CR": "Type-Ia",
    "Calb1": "Type-Ib",
    "Lypd1": "Type-Ic",
    "Prph": "Type-II",
}


def check_processing_status():
    s3 = create_s3_target()

    # For checking the dataset names.
    # content = s3.open(f"{BUCKET_NAME}/project.json", mode="r", encoding="utf-8")
    # info = json.loads(content.read())
    # datasets = info["datasets"]
    # for name in datasets:
    #     print(name)
    # breakpoint()

    for cochlea, channels in COCHLEAE_FOR_SUBTYPES.items():
        try:
            content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        except FileNotFoundError:
            print(cochlea, "is not yet on MoBIE")
            print()
            continue
        info = json.loads(content.read())
        sources = info["sources"]

        channels_found = [name for name in channels if name in sources]
        channels_missing = [name for name in channels if name not in sources]

        print("Cochlea:", cochlea)
        print("Found the expected channels:", channels_found)
        if channels_missing:
            print("Missing the expected channels:", channels_missing)

        if "SGN_v2" in sources:
            print("SGN segmentation is present with name SGN_v2")
        elif "PV_SGN_v2" in sources:
            print("SGN segmentation is present with name PV_SGN_v2")
        else:
            print("SGN segmentation is MISSING")
        print()


def analyze_subtypes_intensity_based():
    s3 = create_s3_target()
    seg_name = "PV_SGN_v2"

    threshold_dict = {}
    output_folder = "./subtype_analysis"
    os.makedirs(output_folder, exist_ok=True)

    for cochlea, channels in COCHLEAE_FOR_SUBTYPES.items():
        # Remove the PV channel, which we don't need for analysis.
        channels = channels[1:]

        # FIXME
        if cochlea != "M_LR_000099_L":
            continue

        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        sources = info["sources"]

        # Load the segmentation table.
        seg_source = sources[seg_name]
        table_folder = os.path.join(
            BUCKET_NAME, cochlea, seg_source["segmentation"]["tableData"]["tsv"]["relativePath"]
        )
        table_content = s3.open(os.path.join(table_folder, "default.tsv"), mode="rb")
        table = pd.read_csv(table_content, sep="\t")

        # Get the SGNs in the main component
        table = table[table.component_labels == 1]
        valid_sgns = table.label_id

        output_table = {"label_id": table.label_id.values}
        threshold_dict[cochlea] = {}

        # Analyze the different channels (= different subtypes).
        for channel in channels:
            # Load the intensity table.
            intensity_path = os.path.join(table_folder, f"{channel}_PV-SGN-v2_object-measures.tsv")
            table_content = s3.open(intensity_path, mode="rb")
            intensities = pd.read_csv(table_content, sep="\t")
            intensities = intensities[intensities.label_id.isin(valid_sgns)]
            assert len(table) == len(intensities)
            assert (intensities.label_id.values == table.label_id.values).all()

            # Intensity based analysis.
            medians = intensities["median"].values

            # TODO: we need to determine the threshold in a better way / validate it in MoBIE.
            intensity_threshold = float(threshold_otsu(medians))
            threshold_dict[cochlea][channel] = intensity_threshold

            subtype = CHANNEL_TO_TYPE[channel]
            output_table[f"{channel}_median"] = medians
            output_table[f"is_{subtype}"] = medians > intensity_threshold

        # Add the frequency mapping.
        # TODO

        out_path = os.path.join(output_folder, f"{cochlea}_subtype_analysis.tsv")
        output_table = pd.DataFrame(output_table)
        output_table.to_csv(out_path, sep="\t")

    threshold_out = os.path.join(output_folder, "thresholds.json")
    with open(threshold_out, "w") as f:
        json.dump(threshold_dict, f, sort_keys=True, indent=4)


# General notes:
# M_LR_000099_L: PV looks weird and segmentation doesn't work so well. Besides this intensities look good.
#                Double check if this is the right channel. Maybe we try domain adaptation here?
# M_LR_000214_L: PV looks correct, segmentation is not there yet.
# M_AMD_N62_L: PV signal and segmentation look good.
# M_AMD_Runx1_L: PV looks a bit off, but should work. Segmentation is not there yet.
def main():
    # check_processing_status()
    analyze_subtypes_intensity_based()


if __name__ == "__main__":
    main()
