import pandas as pd
import numpy as np

# Directory with synapse measurement tables
SYNAPSE_DIR_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/synapses"
# SYNAPSE_DIR_ROOT = "./synapses"


# Define the animal specific octave bands.
def _get_mapping(animal):
    if animal == "mouse":
        bin_edges = [0, 2, 4, 8, 16, 32, 64, np.inf]
        bin_labels = [
            "<2 k", "2–4 k", "4–8 k", "8–16 k", "16–32 k", "32–64 k", ">64 k"
        ]
    elif animal == "gerbil":
        bin_edges = [0, 0.5, 1, 2, 4, 8, 16, 32, np.inf]
        bin_labels = [
            "<0.5 k", "0.5–1 k", "1–2 k", "2–4 k", "4–8 k", "8–16 k", "16–32 k", ">32 k"
        ]
    else:
        raise ValueError
    assert len(bin_edges) == len(bin_labels) + 1
    return bin_edges, bin_labels


def frequency_mapping(frequencies, values, animal="mouse", transduction_efficiency=False):
    # Get the mapping of frequencies to octave bands for the given species.
    bin_edges, bin_labels = _get_mapping(animal)

    # Construct the data frame with octave bands.
    df = pd.DataFrame({"freq_khz": frequencies, "value": values})
    df["octave_band"] = pd.cut(
        df["freq_khz"], bins=bin_edges, labels=bin_labels, right=False
    )

    if transduction_efficiency:  # We compute the transduction efficiency per band.
        num_pos = df[df["value"] == 1].groupby("octave_band", observed=False).size()
        num_tot = df[df["value"].isin([1, 2])].groupby("octave_band", observed=False).size()
        value_by_band = (num_pos / num_tot).reindex(bin_labels)
    else:  # Otherwise, aggregate the values over the octave band using the mean.
        value_by_band = (
            df.groupby("octave_band", observed=True)["value"]
              .mean()
              .reindex(bin_labels)   # keep octave order even if a bin is empty
        )
    return value_by_band


def sliding_runlength_sum(run_length, values, width):
    assert len(run_length) == len(values)
    # Create data frame and sort it.
    df = pd.DataFrame({"run_length": run_length, "value": values})
    df = df.sort_values("run_length").reset_index(drop=True).copy()

    x = df["run_length"].to_numpy()
    y = df["value"].to_numpy()

    cumsum = np.cumsum(y)
    start_idx = np.searchsorted(x, x - width, side="left")
    window_sum = cumsum - np.concatenate(([0], cumsum[:-1]))[start_idx]
    assert len(window_sum) == len(x)

    return x, window_sum


# For mouse
def literature_reference_values(structure):
    if structure == "SGN":
        lower_bound, upper_bound = 9141, 11736
    elif structure == "IHC":
        lower_bound, upper_bound = 656, 681
    elif structure == "synapse":
        lower_bound, upper_bound = 9.1, 20.7
    else:
        raise ValueError
    return lower_bound, upper_bound


# For gerbil
def literature_reference_values_gerbil(structure):
    if structure == "SGN":
        lower_bound, upper_bound = 24700, 28450
    elif structure == "IHC":
        lower_bound, upper_bound = 1081, 1081
    elif structure == "synapse":
        lower_bound, upper_bound = 12.5, 25
    else:
        raise ValueError
    return lower_bound, upper_bound


def to_alias(cochlea_name):
    name_short = cochlea_name.replace("_", "").replace("0", "")
    name_to_alias = {
        "MLR226L": "M01L",
        "MLR226R": "M01R",
        "MLR227L": "M02L",
        "MLR227R": "M02R",
    }
    return name_to_alias[name_short]
