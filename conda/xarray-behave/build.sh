# We need to turn pip index back on because Anaconda turns
# it off for some reason.
export PIP_NO_INDEX=False
export PIP_NO_DEPENDENCIES=False
export PIP_IGNORE_INSTALLED=False

if [[ "$OSTYPE" == "darwin"* ]]; then
  $PYTHON -m pip install simpleaudio -vv --no-dependencies
fi
$PYTHON -m pip install pkgutil_resolve_name -vv --no-dependencies
$PYTHON -m pip install xarray-behave[gui] -vv --no-dependencies
