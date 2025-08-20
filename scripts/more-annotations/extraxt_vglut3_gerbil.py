import os
import json
from flamingo_tools.extract_block_util import extract_block


blocks_to_annotate = [
    "[1157.0001356895104,1758.4586038866773,994.1494008046312]",
    "[1257.4854619519856,1712.418054399143,942.993234707371]",
    "[1329.892068232878,1421.8487165158099,712.9291398862247]",
    "[1035.3286672774282,1844.919679510697,826.5595378176982]"
]

empty_blocks = [
    "[1066.8140315229548,1654.678601097876,994.1494008046308]",
    "[1372.9314226188667,1698.1843589090392,805.5965454893357]",
    "[1079.515087933512,1425.90033123735,1006.1353228190363]",
    "[825.208612674945,565.9088211207202,1170.995860154057]",
]


input_path = "G_EK_000233_L/images/ome-zarr/Vglut3.ome.zarr"
input_key = "s0"
output_key = None
resolution = 0.38

output_folder = "./G_EK_000233_L_VGlut3"
os.makedirs(output_folder, exist_ok=True)
for coords in blocks_to_annotate:
    halo = [196, 196, 48]
    coords = json.loads(coords)
    extract_block(
        input_path, coords, output_folder, input_key, output_key, resolution, halo,
        tif=True, s3=True,
    )

output_folder = "./G_EK_000233_L_VGlut3_empty"
os.makedirs(output_folder, exist_ok=True)
for coords in empty_blocks:
    coords = json.loads(coords)
    extract_block(
        input_path, coords, output_folder, input_key, output_key, resolution, halo,
        tif=True, s3=True,
    )
