import json
import os

import pandas as pd
from skimage.filters import threshold_otsu

from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target, get_s3_path
from flamingo_tools.measurements import compute_object_measures

# Map from cochlea names to channels
COCHLEAE_FOR_SUBTYPES = {
    "M_LR_000099_L": ["PV", "Calb1", "Lypd1"],
    "M_LR_000214_L": ["PV", "CR", "Calb1"],
    "M_AMD_N62_L": ["PV", "CR", "Calb1"],
    "M_AMD_N180_R": ["CR", "Ntng1", "CTBP2"],
    # Mutant / some stuff is weird.
    # "M_AMD_Runx1_L": ["PV", "Lypd1", "Calb1"],
    # This one still has to be stitched:
    # "M_LR_000184_R": {"PV", "Prph"},
    # We don't have PV here, so we exclude these two for now.
    # "M_AMD_00N180_L": {"CR", "Ntng1", "Lypd1"},
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
    "Ntng1": "Type-Ib/c",
}

# For custom thresholds.
THRESHOLDS = {
    "M_LR_000214_L": {
    },
    "M_AMD_N62_L": {
    },
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

    missing_tables = {}

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
            seg_name = "SGN-v2"
            table_folder = "tables/SGN_v2"
        elif "PV_SGN_v2" in sources:
            print("SGN segmentation is present with name PV_SGN_v2")
            seg_name = "PV-SGN-v2"
            table_folder = "tables/PV_SGN_v2"
        elif "CR_SGN_v2" in sources:
            print("SGN segmentation is present with name CR_SGN_v2")
            seg_name = "CR-SGN-v2"
            table_folder = "tables/CR_SGN_v2"
        else:
            print("SGN segmentation is MISSING")
            print()
            continue

        # Check which tables we have.
        expected_tables = [f"{chan}_{seg_name}_object-measures.tsv" for chan in channels]
        tables = s3.ls(os.path.join(BUCKET_NAME, cochlea, table_folder))
        tables = [os.path.basename(tab) for tab in tables]

        this_missing_tables = []
        for exp_tab in expected_tables:
            if exp_tab not in tables:
                print("Missing table:", exp_tab)
                this_missing_tables.append(exp_tab)
        missing_tables[cochlea] = this_missing_tables
        print()

    return missing_tables


def require_missing_tables(missing_tables):
    output_root = "./object_measurements"

    for cochlea, missing_tabs in missing_tables.items():
        seg_name = "PV_SGN_v2" if "PV" in COCHLEAE_FOR_SUBTYPES[cochlea] else "CR_SGN_v2"
        for missing in missing_tabs:
            channel = missing.split("_")[0]
            print(cochlea, channel)

            img_s3 = f"{cochlea}/images/ome-zarr/{channel}.ome.zarr"
            seg_s3 = f"{cochlea}/images/ome-zarr/{seg_name}.ome.zarr"
            seg_table_s3 = f"{cochlea}/tables/{seg_name}/default.tsv"
            img_path, _ = get_s3_path(img_s3)
            seg_path, _ = get_s3_path(seg_s3)

            output_folder = os.path.join(output_root, cochlea)
            os.makedirs(output_folder, exist_ok=True)
            output_table_path = os.path.join(output_folder, f"{channel}_{seg_name}_object-measures.tsv")
            compute_object_measures(
                image_path=img_path,
                segmentation_path=seg_path,
                segmentation_table_path=seg_table_s3,
                output_table_path=output_table_path,
                image_key="s0",
                segmentation_key="s0",
                s3_flag=True,
                component_list=[1],
                n_threads=8,
            )
            return

            # TODO S3 upload


def get_data_for_subtype_analysis():
    s3 = create_s3_target()

    threshold_dict = {}
    output_folder = "./subtype_analysis"
    os.makedirs(output_folder, exist_ok=True)

    for cochlea, channels in COCHLEAE_FOR_SUBTYPES.items():
        if "PV" in channels:
            reference_channel = "PV"
            seg_name = "PV_SGN_v2"
        else:
            assert "CR" in channels
            reference_channel = "CR"
            seg_name = "CR_SGN_v2"
        reference_channel, seg_name

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
            try:
                table_content = s3.open(intensity_path, mode="rb")
            except FileNotFoundError:
                print(intensity_path, "is missing")
                continue
            intensities = pd.read_csv(table_content, sep="\t")
            intensities = intensities[intensities.label_id.isin(valid_sgns)]
            assert len(table) == len(intensities)
            assert (intensities.label_id.values == table.label_id.values).all()

            # Intensity based analysis.
            medians = intensities["median"].values

            # TODO: we need to determine the threshold in a better way / validate it in MoBIE.
            intensity_threshold = THRESHOLDS.get(cochlea, {}).get(channel, None)
            if intensity_threshold is None:
                print("Could not find a threshold for", cochlea, channel, "falling back to OTSU")
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
# M_AMD_N180_R: Need SGN segmentation based on CR.
def main():
    missing_tables = check_processing_status()
    require_missing_tables(missing_tables)

    # analyze_subtypes_intensity_based()


if __name__ == "__main__":
    main()
