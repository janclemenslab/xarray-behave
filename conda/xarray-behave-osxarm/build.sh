# We need to turn pip index back on because Anaconda turns
# it off for some reason.
export PIP_NO_INDEX=False
export PIP_NO_DEPENDENCIES=False
export PIP_IGNORE_INSTALLED=False

$PYTHON -m pip install pysoundfile -vv
$PYTHON -m pip install pyside6 pyqtgraph qtpy -vv
$PYTHON -m pip install opencv-python-headless -vv
$PYTHON -m pip install pyvideoreader -vv --no-dependencies
$PYTHON -m pip install xarray-behave[gui] -vv --no-dependencies
