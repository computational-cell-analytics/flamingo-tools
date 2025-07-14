import argparse

import imageio.v3 as imageio
import napari
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from magicgui import magicgui

from elf.parallel.distance_transform import distance_transform
from elf.parallel.seeded_watershed import seeded_watershed

from flamingo_tools.measurements import compute_object_measures_impl


class HistogramWidget(QWidget):
    """Qt widget that draws/updates a histogram for one napari layer."""
    def __init__(self, statistics, default_stat, bins: int = 32, parent=None):
        super().__init__(parent)
        self.bins = bins

        # --- layout ------------------------------------------------------
        self.fig, self.ax = plt.subplots(figsize=(4, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)

        # We exclude the label id and the volume / surface measurements.
        self.stat_names = statistics.columns[1:-2] if len(statistics.columns) > 2 else statistics.columns[1:]
        self.param_choices = self.stat_names

        self.param_box = QComboBox()
        self.param_box.addItems(self.param_choices)
        self.param_box.setCurrentText(default_stat)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.update_hist)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Choose statistic:"))
        layout.addWidget(self.param_box)
        layout.addWidget(self.canvas)
        layout.addWidget(self.refresh_btn)
        self.setLayout(layout)

        self.statistics = statistics
        self.update_hist()  # initial draw

    def update_hist(self):
        """Redraw the histogram."""
        self.ax.clear()

        stat_name = self.param_box.currentText()

        data = self.statistics[stat_name]
        # Seaborn version (nicer aesthetics)
        sns.histplot(data, bins=self.bins, ax=self.ax, kde=False)

        self.ax.set_xlabel(f"{stat_name} GFP Intensity")
        self.ax.set_ylabel("Count")
        self.canvas.draw_idle()


def _create_stat_widget(statistics, default_stat):
    widget = HistogramWidget(statistics, default_stat)
    return widget


# Extend via watershed, this could work for a better alignment.
def _extend_sgns_complex(gfp, sgns):
    # 1.) compute distance to the SGNs to create the background seed.
    print("Compute edt")

    # Could use parallel impl
    distance_threshol = 7
    distances = distance_transform(sgns == 0)

    # Erode seeds?
    seeds = sgns.copy()
    bg_seed_id = int(seeds.max()) + 1
    seeds[distances > distance_threshol] = bg_seed_id

    # Dilate to cover everything on the boundary?
    print("Run watershed")
    sgns_extended = seeded_watershed(gfp, markers=seeds)
    sgns_extended[sgns_extended == bg_seed_id] = 0

    v = napari.Viewer()
    v.add_image(gfp)
    v.add_image(distances)
    v.add_labels(seeds)
    v.add_labels(sgns_extended)
    napari.run()


# Just dilate by 3 pixels.
def _extend_sgns_simple(gfp, sgns, dilation):
    block_shape = (128,) * 3
    halo = (dilation + 2,) * 3

    distances = distance_transform(sgns == 0, block_shape=block_shape, halo=halo, n_threads=8)
    mask = distances < dilation

    sgns_extended = np.zeros_like(sgns)
    sgns_extended = seeded_watershed(
        distances, sgns, sgns_extended, block_shape=block_shape, halo=halo, n_threads=8, mask=mask
    )

    return sgns_extended


def _create_mask(sgns_extended, gfp):
    from skimage.transform import downscale_local_mean, resize

    gfp_averaged = downscale_local_mean(gfp, factors=(16, 16, 16))
    # The 35th percentile seems to be a decent approximation for the background subtraction.
    threshold = np.percentile(gfp_averaged, 35)
    mask = gfp_averaged > threshold
    mask = resize(mask, sgns_extended.shape, order=0, anti_aliasing=False, preserve_range=True).astype(bool)
    mask[sgns_extended != 0] = 0

    # v = napari.Viewer()
    # v.add_image(gfp)
    # v.add_image(gfp_averaged, scale=(16, 16, 16))
    # v.add_labels(mask)
    # # v.add_labels(mask, scale=(16, 16, 16))
    # v.add_labels(sgns_extended)
    # napari.run()

    return mask


def gfp_annotation(prefix, default_stat="median", background_norm=None):
    assert background_norm in (None, "division", "subtraction")

    gfp = imageio.imread(f"{prefix}_GFP_resized.tif")
    sgns = imageio.imread(f"{prefix}_SGN_resized_v2.tif")
    pv = imageio.imread(f"{prefix}_PV_resized.tif")

    # bb = np.s_[128:-128, 128:-128, 128:-128]
    # gfp, sgns, pv = gfp[bb], sgns[bb], pv[bb]
    # print(gfp.shape)

    # Extend the sgns so that they cover the SGN boundaries.
    # sgns_extended = _extend_sgns(gfp, sgns)
    # TODO we need to integrate this directly in the object measurement to efficiently do it at scale.
    sgns_extended = _extend_sgns_simple(gfp, sgns, dilation=4)

    # Compute the intensity statistics.
    if background_norm is None:
        mask = None
        feature_set = "default"
    else:
        mask = _create_mask(sgns_extended, gfp)
        assert mask.shape == sgns_extended.shape
        feature_set = "default_background_norm" if background_norm == "division" else "default_background_subtract"
    statistics = compute_object_measures_impl(
        gfp, sgns_extended, feature_set=feature_set, foreground_mask=mask, median_only=True
    )

    # Open the napari viewer.
    v = napari.Viewer()

    # Add the base layers.
    v.add_image(gfp, name="GFP")
    v.add_image(pv, visible=False, name="PV")
    v.add_labels(sgns, visible=False, name="SGNs")
    v.add_labels(sgns_extended, name="SGNs-extended")
    if mask is not None:
        v.add_labels(mask, name="mask-for-background", visible=False)

    # Add additional layers for intensity coloring and classification
    # data_numerical = np.zeros(gfp.shape, dtype="float32")
    data_labels = np.zeros(gfp.shape, dtype="uint8")

    # v.add_image(data_numerical, name="gfp-intensity")
    v.add_labels(data_labels, name="positive-negative")

    # Add widgets:

    # 1.) The widget for selcting the statistics to be used and displaying the histogram.
    stat_widget = _create_stat_widget(statistics, default_stat)

    # 2.) Precompute statistic ranges.
    stat_names = stat_widget.stat_names
    all_values = statistics[stat_names].values
    min_val = all_values.min()
    max_val = all_values.max()

    # 3.) The widget for printing the intensity of a selected cell.
    @magicgui(
        value={"label": "value", "enabled": False, "widget_type": "FloatSpinBox", "min": 0.0, "max": max_val},
        call_button="Pick Value"
    )
    def pick_widget(viewer: napari.Viewer, value: float = 0.0):
        layer = viewer.layers["SGNs-extended"]
        selected_id = layer.selected_label

        stat_name = stat_widget.param_box.currentText()
        label_ids = statistics.label_id.values
        if selected_id not in label_ids:
            return {"value": 0.0}

        vals = statistics[stat_name].values
        picked_value = vals[label_ids == selected_id][0]
        pick_widget.value.value = picked_value

    # 4.) The widget for setting the threshold and updating the positive / negative classification based on it.
    @magicgui(
        threshold={
            "widget_type": "FloatSlider",
            "label": "Threshold",
            "min": min_val,
            "max": max_val,
            "step": 1,
        },
        call_button="Apply",
    )
    def threshold_widget(viewer: napari.Viewer, threshold: float = (max_val - min_val) / 2):
        label_ids = statistics.label_id.values
        stat_name = stat_widget.param_box.currentText()
        vals = statistics[stat_name].values
        pos_ids = label_ids[vals >= threshold]
        neg_ids = label_ids[vals <= threshold]
        data_labels = np.zeros(gfp.shape, dtype="uint8")
        data_labels[np.isin(sgns_extended, pos_ids)] = 2
        data_labels[np.isin(sgns_extended, neg_ids)] = 1
        viewer.layers["positive-negative"].data = data_labels

    threshold_widget.viewer.value = v

    # Bind the widgets.
    v.window.add_dock_widget(stat_widget, area="right")
    v.window.add_dock_widget(pick_widget, area="right")
    v.window.add_dock_widget(threshold_widget, area="right")
    stat_widget.setWindowTitle("GFP Histogram")

    napari.run()


# Cochlea chanel registration quality:
# - M_LR_000144_L: rough alignment is ok, but specific alignment is a bit poor.
# - M_LR_000145_L: rough alignment is ok, detailed alignment also ok.
# - M_LR_000151_R: rough alignment is ok, detailed alignment also ok.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("-b", "--background_norm")
    args = parser.parse_args()

    gfp_annotation(args.prefix, background_norm=args.background_norm)


if __name__ == "__main__":
    main()
