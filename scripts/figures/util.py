import pandas as pd
import numpy as np


# Define the animal specific octave bands.
def _get_mapping(animal):
    if animal == "mouse":
        bin_edges = [0, 1, 2, 4, 8, 16, 32, 64, np.inf]
        bin_labels = [
            "<1 k", "1–2 k", "2–4 k", "4–8 k", "8–16 k", "16–32 k", "32–64 k", ">64 k"
        ]
    elif animal == "gerbil":
        bin_edges = [0, 0.5, 1, 2, 4, 8, 16, 32, np.inf]
        bin_labels = [
            "<0.5 k", "0.5–1 k", "1–2 k", "2–4 k", "4–8 k", "8–16 k", "16–32 k", ">32 k"
        ]
    else:
        raise ValueError
    assert len(bin_edges) == len(bin_labels)
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

# mean_by_band.plot.bar()
# plt.ylabel("Mean value")
# plt.xlabel("Octave band (kHz)")
# plt.title("Mouse data binned by octave")
# plt.tight_layout()
# plt.show()
#
