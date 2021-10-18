# We need to turn pip index back on because Anaconda turns
# it off for some reason.
export PIP_NO_INDEX=False
export PIP_NO_DEPENDENCIES=False
export PIP_IGNORE_INSTALLED=False

$PYTHON -m pip install pysoundfile -vv
$PYTHON -m pip install xarray-behave[gui] -vv --no-dependencies
