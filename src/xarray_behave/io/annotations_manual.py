import h5py
import logging
import numpy as np
import pandas as pd
import scipy.io
from typing import Optional
from .. import io, annot, xarray_behave, event_utils

logger = logging.getLogger(__name__)


@io.register_provider
class Manual_xb_csv(io.BaseProvider):
    KIND = "annotations_manual"
    NAME = "XB csv"
    SUFFIXES = ["_annotations.csv", "_songmanual.csv", ".csv"]

    def load(self, filename: Optional[str] = None):
        """Load output produced by xb."""
        df = pd.read_csv(filename)
        if not all([item in df.columns for item in ["name", "start_seconds", "stop_seconds"]]):
            logger.error(
                f"Malformed CSV file {filename} - needs to have these columns: ['name','start_seconds', 'stop_seconds']. Returning empty results"
            )
            event_seconds = annot.Events()  # make empty
        else:
            event_seconds = annot.Events.from_df(df)
        return event_seconds, event_seconds.categories


@io.register_provider
class Definitions(io.BaseProvider):
    KIND = "definitions_manual"
    NAME = "XB def csv"
    SUFFIXES = ["_definitions.csv"]

    def load(self, filename: Optional[str] = None):
        """Load output produced by xb."""
        # load definitions and add to annot instance
        if filename is None:
            filename = self.path

        definitions = np.loadtxt(filename, dtype=str, delimiter=",")
        if definitions.ndim == 0:
            definitions = [[str(definitions)]]
        elif definitions.ndim == 1:
            if len(definitions) == 2 and definitions[1] in ("event", "segment"):
                definitions = [definitions.tolist()]
            else:
                definitions = [[item] for item in definitions.tolist()]

        event_seconds = annot.Events()  # make empty
        for definition in definitions:
            event_seconds.add_name(name=definition[0], category="event")

        return event_seconds, event_seconds.categories


@io.register_provider
class Manual_xb_zarr(io.BaseProvider):
    KIND = "annotations_manual"
    NAME = "XB zarr"
    SUFFIXES = ["_songmanual.zarr"]

    def load(self, filename: Optional[str] = None):
        """Load output produced by xb (legacy format)."""
        if filename is None:
            filename = self.path

        manual_events_ds = xarray_behave.load(filename)

        if "event_categories" not in manual_events_ds:
            event_categories_list = event_utils.infer_event_categories_from_traces(manual_events_ds.song_events.data)

            manual_events_ds = manual_events_ds.assign_coords({"event_categories": (("event_types"), event_categories_list)})

        event_seconds = event_utils.detect_events(manual_events_ds)

        event_categories = {}
        for typ in manual_events_ds.event_types.data:
            event_categories[typ] = "event"

        return event_seconds, event_categories


@io.register_provider
class Manual_matlab(io.BaseProvider):
    KIND = "annotations_manual"
    NAME = "FSS matlab"
    SUFFIXES = ["_songmanual.mat"]

    def load(self, filename: Optional[str] = None):
        """Load output produced by the matlab ManualSegmenter."""
        if filename is None:
            filename = self.path

        try:
            mat_data = scipy.io.loadmat(filename)
        except NotImplementedError:
            with h5py.File(filename, "r") as f:
                mat_data = dict()
                for key, val in f.items():
                    mat_data[key.lower()] = val[:].T

        events_seconds = dict()
        event_categories = dict()
        for key, val in mat_data.items():
            if len(val) and hasattr(val, "ndim") and val.ndim == 2 and not key.startswith("_"):  # ignore matfile metadata
                events_seconds[key.lower() + "_manual"] = np.sort(val[:, 1:])
                event_categories[key.lower() + "_manual"] = "event"
        return events_seconds, event_categories
