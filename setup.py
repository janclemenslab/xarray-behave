from setuptools import setup, find_packages
import codecs
import re
import os

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='xarray_behave',
      version=find_version("src/xarray_behave/__init__.py"),
      description='xarray_behave',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/janclemenslab/xarray-behave',
      author='Jan Clemens',
      author_email='clemensjan@googlemail.com',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      package_data={'xarray_behave': ['gui/forms/*']},
      install_requires=['numpy', 'scipy', 'xarray', 'h5py', 'zarr', 'flammkuchen',
                        'dask', 'toolz', 'samplestamps', 'soundfile'],
      extras_require={'gui': ['pyside2', 'pyqtgraph==0.11.0rc0', 'pyvideoreader',
                              'sounddevice', 'scikit-image', 'opencv-python', 'pyyaml', 'defopt']},
      include_package_data=True,
      zip_safe=False,
      entry_points={'console_scripts': ['xb=xarray_behave.gui.app:cli'], }
     )
