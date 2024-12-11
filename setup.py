import runpy
from setuptools import setup, find_packages

version = runpy.run_path("flamingo_tools/version.py")["__version__"]
setup(
    name="flamingo_tools",
    packages=find_packages(exclude=["test"]),
    version=version,
    author="Constantin Pape",
    license="MIT",
    entry_points={
        "console_scripts": [
            "convert_flamingo = flamingo_tools.data_conversion:convert_lightsheet_to_bdv_cli"
        ]
    }
)
