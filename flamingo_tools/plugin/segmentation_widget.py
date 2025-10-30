import copy
import re
from typing import Optional, Union

import napari
import torch

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox

from .base_widget import BaseWidget
from .util import get_default_tiling, get_device
from ..model_utils import get_model, get_model_registry


def _load_custom_model(model_path: str, device: Optional[Union[str, torch.device]] = None) -> torch.nn.Module:
    model_path = _clean_filepath(model_path)
    if device is None:
        device = get_device(device)
    try:
        model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    except Exception as e:
        print(e)
        print("model path", model_path)
        return None
    return model


def _available_devices():
    available_devices = []
    for i in ["cuda", "mps", "cpu"]:
        try:
            device = get_device(i)
        except RuntimeError:
            pass
        else:
            available_devices.append(device)
    return available_devices


def _get_current_tiling(tiling: dict, default_tiling: dict, model_type: str):
    # get tiling values from qt objects
    for k, v in tiling.items():
        for k2, v2 in v.items():
            if isinstance(v2, int):
                continue
            elif hasattr(v2, "value"):  # If it's a QSpinBox, extract the value
                tiling[k][k2] = v2.value()
            else:
                raise TypeError(f"Unexpected type for tiling value: {type(v2)} at {k}/{k2}")
    show_info(f"Using tiling: {tiling}")
    return tiling


def _clean_filepath(filepath):
    """Cleans a given filepath by:
    - Removing newline characters (\n)
    - Removing escape sequences
    - Stripping the 'file://' prefix if present

    Args:
        filepath (str): The original filepath

    Returns:
        str: The cleaned filepath
    """
    # Remove 'file://' prefix if present
    if filepath.startswith("file://"):
        filepath = filepath[7:]

    # Remove escape sequences and newlines
    filepath = re.sub(r'\\.', '', filepath)
    filepath = filepath.replace('\n', '').replace('\r', '')

    return filepath


def _run_segmentation(image, model, model_type, tiling):
    # return segmentation
    pass


class SegmentationWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()
        self.tiling = {}

        # Create the image selection dropdown.
        self.image_selector_name = "Image data"
        self.image_selector_widget = self._create_layer_selector(self.image_selector_name, layer_type="Image")

        # Create buttons and widgets.
        self.predict_button = QPushButton("Run Segmentation")
        self.predict_button.clicked.connect(self.on_predict)
        self.model_selector_widget = self.load_model_widget()
        self.settings = self._create_settings_widget()

        # Add the widgets to the layout.
        layout.addWidget(self.image_selector_widget)
        layout.addWidget(self.model_selector_widget)
        layout.addWidget(self.settings)
        layout.addWidget(self.predict_button)

        self.setLayout(layout)

    def load_model_widget(self):
        model_widget = QWidget()
        title_label = QLabel("Select Model:")

        # Exclude the models that are only offered through the CLI and not in the plugin.
        model_list = list(get_model_registry().urls.keys())

        models = ["- choose -"] + model_list
        self.model_selector = QComboBox()
        self.model_selector.addItems(models)
        # Create a layout and add the title label and combo box
        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(self.model_selector)

        # Set layout on the model widget
        model_widget.setLayout(layout)
        return model_widget

    def on_predict(self):
        # Get the model and postprocessing settings.
        model_type = self.model_selector.currentText()
        custom_model_path = self.checkpoint_param.text()
        if model_type == "- choose -" and custom_model_path is None:
            show_info("INFO: Please choose a model.")
            return

        device = get_device(self.device_dropdown.currentText())

        # Load the model. Override if user chose custom model
        if custom_model_path:
            model = _load_custom_model(custom_model_path, device)
            if model:
                show_info(f"INFO: Using custom model from path: {custom_model_path}")
                model_type = "custom"
            else:
                show_info(f"ERROR: Failed to load custom model from path: {custom_model_path}")
                return
        else:
            model = get_model(model_type, device)

        # Get the image data.
        image = self._get_layer_selector_data(self.image_selector_name)
        if image is None:
            show_info("INFO: Please choose an image.")
            return

        # Get the current tiling.
        self.tiling = _get_current_tiling(self.tiling, self.default_tiling, model_type)
        segmentation = _run_segmentation(image, model=model, model_type=model_type, tiling=self.tiling)

        self.viewer.add_labels(segmentation, name=model_type)
        show_info(f"INFO: Segmentation of {model_type} added to layers.")

    def _create_settings_widget(self):
        setting_values = QWidget()
        # setting_values.setToolTip(get_tooltip("embedding", "settings"))
        setting_values.setLayout(QVBoxLayout())

        # Create UI for the device.
        device = "auto"
        device_options = ["auto"] + _available_devices()

        self.device_dropdown, layout = self._add_choice_param("device", device, device_options)
        setting_values.layout().addLayout(layout)

        # Create UI for the tile shape.
        self.default_tiling = get_default_tiling()
        self.tiling = copy.deepcopy(self.default_tiling)
        self.tiling["tile"]["x"], self.tiling["tile"]["y"], self.tiling["tile"]["z"], layout = self._add_shape_param(
            ("tile_x", "tile_y", "tile_z"),
            (self.default_tiling["tile"]["x"], self.default_tiling["tile"]["y"], self.default_tiling["tile"]["z"]),
            min_val=0, max_val=2048, step=16,
            # tooltip=get_tooltip("embedding", "tiling")
        )
        setting_values.layout().addLayout(layout)

        # Create UI for the halo.
        self.tiling["halo"]["x"], self.tiling["halo"]["y"], self.tiling["halo"]["z"], layout = self._add_shape_param(
            ("halo_x", "halo_y", "halo_z"),
            (self.default_tiling["halo"]["x"], self.default_tiling["halo"]["y"], self.default_tiling["halo"]["z"]),
            min_val=0, max_val=512,
            # tooltip=get_tooltip("embedding", "halo")
        )
        setting_values.layout().addLayout(layout)

        self.checkpoint_param, layout = self._add_string_param(
            name="checkpoint", value="", title="Load Custom Model",
            placeholder="path/to/checkpoint.pt",
        )
        setting_values.layout().addLayout(layout)

        settings = self._make_collapsible(widget=setting_values, title="Advanced Settings")
        return settings
