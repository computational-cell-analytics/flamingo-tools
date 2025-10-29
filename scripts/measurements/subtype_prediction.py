import os
from glob import glob
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import pandas as pd

from skimage.measure import regionprops
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT_AMD = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/SGN_subtypes/Result_AMD"
ROOT_EK = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/SGN_subtypes/Result_EK"


def load_annotations(pattern):
    paths = sorted(glob(pattern))
    annotations = [path[len(pattern):] for path in paths]
    channels = [annotation.split("_")[0] for annotation in annotations]
    return paths, channels


# Get the features and per channel labels from this crop.
def extract_crop_data(crop_table, crop_root):
    table = pd.read_csv(crop_table, sep="\t")
    prefix = Path(crop_table).stem

    # Get the paths to all annotations.
    paths_amd, channels_amd = load_annotations(os.path.join(ROOT_AMD, f"positive-negative_{prefix}*"))
    paths_ek, channels_ek = load_annotations(os.path.join(ROOT_EK, f"positive-negative_{prefix}*"))
    channel_names = list(set(channels_amd))

    # Load the segmentation.
    seg_path = os.path.join(crop_root, f"{prefix}_PV_SGN_v2.tif")
    seg = imageio.imread(seg_path)

    # Load the features (= intensity and PV intensity ratios) for both channels.
    features = table[
        [f"marker_{channel_names[0]}", f"{channel_names[0]}_ratio_PV"] +
        [f"marker_{channel_names[1]}", f"{channel_names[1]}_ratio_PV"]
    ].values

    # Load the labels, derived from the annotations.
    labels = {channel: None for channel in channel_names}
    for channel, path in zip(channels_amd, paths_amd):
        data = imageio.imread(path)
        props = regionprops(seg, data)
        labeling = np.array([prop.max_intensity for prop in props], dtype="int32")
        if labels[channel] is None:
            labels[channel] = labeling
        else:
            # Combine labels so that we only keep the labels that agree, set others to zero
            # (in order to filter them out later).
            prev_labeling = labels[channel]
            disagreement = prev_labeling != labeling
            labeling[disagreement] = 0
            labels[channel] = labeling

    return features, labels


def process_cochlea(cochlea):
    # The root folders for tables and crop data for this cochlea.
    table_root = f"/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/SGN_subtypes/tables_{cochlea}"
    crop_root = f"/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/SGN_subtypes/{cochlea}"

    # Getthe tables for all crops in this cochlea.
    tables = sorted(glob(os.path.join(table_root, "*.tsv")))

    # Iterate over the crops, load the features and the labels per channel.
    features = []
    labels = {}
    for table in tables:
        crop_features, crop_labels = extract_crop_data(table, crop_root)
        features.append(crop_features)
        # Concatenate the labels per channel.
        for channel, labeling in crop_labels.items():
            if channel in labels:
                labels[channel] = np.concatenate([labels[channel], labeling], axis=0)
            else:
                labels[channel] = labeling
    features = np.concatenate(features, axis=0)

    # Train and evaluate logistic regression per channel.
    start, stop = 0, 2
    for channel, labeling in labels.items():
        # Exclude labels with value zero.
        label_mask = labeling != 0
        # Get the features for this channel.
        this_features = features[:, start:stop][label_mask]
        this_labels = labeling[label_mask]

        # Create a train and test split.
        train_features, test_features, train_labels, test_labels = train_test_split(
            this_features, this_labels, test_size=0.3
        )

        # Train and evaluate the classifier.
        classifier = LogisticRegression(penalty="l2")
        classifier.fit(train_features, train_labels)

        prediction = classifier.predict(test_features)
        accuracy = accuracy_score(test_labels, prediction)
        print("Channel:", channel)
        print("Accuracy:", accuracy)

        start += 2
        stop += 2

    # Note: we could do some other things here:
    # - Train a single classifier for subtype prediction (= 4 classes) using all channels.
    # - Use different classifier (e.g. RandomForest); however, accuracy from logistic regression looks fine.
    # - To better understand results we could also look at the confusion matrix.
    # - A better evaluation would be to train and test on separate blocks.

    # The classifier can be saved and loaded with pickle, to apply it to all SGNs in the cochlea later.


def main():
    # Process a cochlea by:
    # - Extracting the features (intensities and intensity ratios) and labels for each crop.
    # - Training a classifier based on the labels and evaluating it.
    process_cochlea("MLR99L")


if __name__ == "__main__":
    main()
