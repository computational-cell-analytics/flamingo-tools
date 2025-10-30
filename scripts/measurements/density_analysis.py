import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flamingo_tools.s3_utils import create_s3_target, BUCKET_NAME


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

    if plot:
        plt.figure(figsize=(6, 3))
        plt.plot(grid, density, lw=2)
        plt.xlabel("Length [µm]")
        plt.ylabel("Density [SGN/µm]")
        plt.tight_layout()
        plt.show()
    else:
        return grid, density


def check_implementation():
    cochlea = "G_EK_000049_L"
    analyze_cochlea(cochlea, plot=True)


def compare_gerbil_cochleae():
    # We need the tonotopic mapping for G_EK_000233_L
    # cochleae = ["G_EK_000233_L", "G_EK_000049_L", "G_EK_000049_R"]
    cochleae = ["G_EK_000049_L", "G_EK_000049_R"]

    plt.figure(figsize=(6, 3))
    for cochlea in cochleae:
        grid, density = analyze_cochlea(cochlea, plot=False)
        plt.plot(grid, density, lw=2, label=cochlea)

    plt.xlabel("Length [µm]")
    plt.ylabel("Density [SGN/µm]")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # check_implementation()
    compare_gerbil_cochleae()


if __name__ == "__main__":
    main()
