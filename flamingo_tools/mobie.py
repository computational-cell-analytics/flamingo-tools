import os
import tempfile
from typing import Tuple

from mobie import add_bdv_image, add_segmentation
from mobie.metadata.dataset_metadata import read_dataset_metadata


# TODO refactor to mobie utils
def _source_exists(mobie_project, mobie_dataset, source_name):
    dataset_folder = os.path.join(mobie_project, mobie_dataset)
    metadata = read_dataset_metadata(dataset_folder)
    sources = metadata.get("sources", {})
    return source_name in sources


def add_raw_to_mobie(
    mobie_project: str,
    mobie_dataset: str,
    source_name: str,
    xml_path: str,
    skip_existing: bool = True,
    setup_id: int = 0,
):
    """
    """
    # Check if we have converted this data already.
    have_source = _source_exists(mobie_project, mobie_dataset, source_name)
    if have_source and skip_existing:
        print(f"Source {source_name} already exists in {mobie_project}:{mobie_dataset}.")
        print("Conversion to mobie will be skipped.")
        return
    elif have_source:
        raise NotImplementedError

    with tempfile.TemporaryDirectory() as tmpdir:
        add_bdv_image(
            xml_path=xml_path,
            root=mobie_project,
            dataset_name=mobie_dataset,
            image_name=source_name,
            tmp_folder=tmpdir,
            file_format="bdv.n5",
            setup_ids=[setup_id],
        )


def add_segmentation_to_mobie(
    mobie_project: str,
    mobie_dataset: str,
    source_name: str,
    segmentation_path: str,
    segmentation_key: str,
    resolution: Tuple[int, int, int],
    unit: str,
    scale_factors: Tuple[Tuple[int, int, int]],
    chunks: Tuple[int, int, int],
    skip_existing: bool = True,
):
    # Check if we have converted this data already.
    have_source = _source_exists(mobie_project, mobie_dataset, source_name)
    if have_source and skip_existing:
        print(f"Source {source_name} already exists in {mobie_project}:{mobie_dataset}.")
        print("Conversion to mobie will be skipped.")
        return
    elif have_source:
        raise NotImplementedError

    with tempfile.TemporaryDirectory() as tmpdir:
        add_segmentation(
            input_path=segmentation_path, input_key=segmentation_key,
            root=mobie_project, dataset_name=mobie_dataset,
            segmentation_name=source_name,
            resolution=resolution, scale_factors=scale_factors,
            chunks=chunks, file_format="bdv.n5",
            tmp_folder=tmpdir
        )
