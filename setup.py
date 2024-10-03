import runpy
from setuptools import setup, find_packages

version = runpy.run_path('flamingo_tools/version.py')['__version__']
setup(name='flamingo_tools',
      packages=find_packages(exclude=['test']),
      version=version,
      author='Constantin Pape',
      license='MIT')
