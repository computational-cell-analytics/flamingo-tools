import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flamingo_tools.s3_utils import create_s3_target, BUCKET_NAME

sys.path.append("../figures")


def _gaussian_kernel1d(sigma_pix, truncate=4.0):
    r = int(truncate * sigma_pix + 0.5)
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-(x**2) / (2 * sigma_pix**2))
    k /= k.sum()
    return k


def density_1d_simple(x, window):
    x = np.asarray(x, dtype=float)
    xmin, xmax = x.min(), x.max()

    # Fixed grid (kept internal for simplicity)
    nbins = 256
    edges = np.linspace(xmin, xmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]

    # Bin counts, smooth with Gaussian, convert to density
    counts, _ = np.histogram(x, bins=edges, density=False)
    sigma_pix = max(window / dx, 1e-9)
    k = _gaussian_kernel1d(sigma_pix)
    counts_smooth = np.convolve(counts, k, mode="same")
    density = counts_smooth / dx

    return centers, density


def open_tsv(fs, path):
    s3_path = os.path.join(BUCKET_NAME, path)
    with fs.open(s3_path, "r") as f:
        table = pd.read_csv(f, sep="\t")
    return table


def analyze_cochlea(cochlea, plot=False):
    fs = create_s3_target()
    seg_name = "SGN_v2"

    table_path = f"tables/{seg_name}/default.tsv"
    table = open_tsv(fs, os.path.join(cochlea, table_path))

    component_ids = [1]
    table = table[table.component_labels.isin(component_ids)]

    window_size = 200.0
    grid, density = density_1d_simple(table["length[µm]"], window=window_size)  # window in same units as x

    if len(grid) != len(density):
        breakpoint()

    if plot:
        plt.figure(figsize=(6, 3))
        plt.plot(grid, density, lw=2)
        plt.xlabel("Length [µm]")
        plt.ylabel("Density [SGN/µm]")
        plt.tight_layout()
        plt.show()
    else:
        return grid, density


def get_sgn_counts(cochlea):
    fs = create_s3_target()
    seg_name = "SGN_v2"

    table_path = f"tables/{seg_name}/default.tsv"
    table = open_tsv(fs, os.path.join(cochlea, table_path))
    component_ids = [1]
    table = table[table.component_labels.isin(component_ids)]

    frequencies = table["frequency[kHz]"].values
    values = np.ones_like(frequencies)

    return frequencies, values


def check_implementation():
    cochlea = "G_EK_000049_L"
    analyze_cochlea(cochlea, plot=True)


def compare_cochleae(cochleae, animal, plot_density=True, plot_tonotopy=True):

    if plot_density:
        plt.figure(figsize=(6, 3))
        for cochlea in cochleae:
            grid, density = analyze_cochlea(cochlea, plot=False)
            plt.plot(grid, density, lw=2, label=cochlea)

        plt.xlabel("Length [µm]")
        plt.ylabel("Density [SGN/µm]")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if plot_tonotopy:
        from util import frequency_mapping

        fig, ax = plt.subplots(figsize=(6, 3))
        for cochlea in cochleae:
            frequencies, values = get_sgn_counts(cochlea)
            sgns_per_band = frequency_mapping(
                frequencies, values, animal=animal, aggregation="sum"
            )
            bin_labels = sgns_per_band.index
            binned_counts = sgns_per_band.values

            band_to_x = {band: i for i, band in enumerate(bin_labels)}
            x_positions = bin_labels.map(band_to_x)
            ax.scatter(x_positions, binned_counts, marker="o", label=cochlea, s=80)

        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel("Octave band [kHz]")
        ax.legend()
        plt.show()


# TODO: implement the same for mouse cochleae (healthy vs. opto treatment)
# also show this in tonotopic mapping


# The visualization has to be improved to make plots understandable.
def main():
    # check_implementation()

    # Comparison for Gerbil.
    # cochleae = ["G_EK_000233_L", "G_EK_000049_L", "G_EK_000049_R"]
    # compare_cochleae(cochleae, animal="gerbil", plot_density=True)

    # Comparison for Mouse.
    # NOTE: There is some problem with M_LR_000143_L and "M_LR_000153_L"
    # I have removed the corresponding pairs for now, but we should investigate and add back.
    cochleae = [
        # Healthy reference cochleae.
        "M_LR_000226_L", "M_LR_000226_R", "M_LR_000227_L", "M_LR_000227_R",
        # Right un-injected cochleae.
        "M_LR_000144_R", "M_LR_000145_R",  "M_LR_000155_R", "M_LR_000189_R",
        # Left injected cochleae.
        "M_LR_000144_L", "M_LR_000145_L", "M_LR_000155_L", "M_LR_000189_L",
    ]
    compare_cochleae(cochleae, animal="mouse")


if __name__ == "__main__":
    main()
