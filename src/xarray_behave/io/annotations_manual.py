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
class RavenPro(io.BaseProvider):
    KIND = "annotations_manual"
    NAME = "Raven Pro selection table"
    SUFFIXES = ["*.selections.txt", "_raven.txt"]

    @classmethod
    def match(cls, filename):
        if not isinstance(filename, str):
            return False
        filename = filename.lower()
        return filename.endswith(".selections.txt") or filename.endswith("_raven.txt")

    def load(self, filename: Optional[str] = None, annotation_column: str = "Annotation"):
        """Load a tab-delimited Raven Pro selection table."""
        if filename is None:
            filename = self.path

        df = pd.read_csv(filename, sep="\t")
        required_columns = ["Begin Time (s)", "End Time (s)", annotation_column]
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Malformed Raven Pro selection table {filename}: missing required columns {missing_columns}.")

        labels = df[annotation_column]
        blank_labels = labels.isna() | labels.astype(str).str.strip().eq("")
        if blank_labels.any():
            rows = (np.flatnonzero(blank_labels.to_numpy()) + 2).tolist()
            raise ValueError(
                f"Malformed Raven Pro selection table {filename}: blank {annotation_column!r} values on rows {rows}."
            )
        labels = labels.astype(str).str.strip()

        begin_seconds = self._numeric_column(df, "Begin Time (s)", filename)
        end_seconds = self._numeric_column(df, "End Time (s)", filename)
        invalid_times = (begin_seconds < 0) | (end_seconds < begin_seconds)
        if invalid_times.any():
            rows = (np.flatnonzero(invalid_times) + 2).tolist()
            raise ValueError(
                f"Malformed Raven Pro selection table {filename}: times must satisfy 0 <= begin <= end on rows {rows}."
            )

        if "Channel" in df.columns:
            raven_channels = self._numeric_column(df, "Channel", filename)
            invalid_channels = (raven_channels < 1) | (raven_channels != np.floor(raven_channels))
            if invalid_channels.any():
                rows = (np.flatnonzero(invalid_channels) + 2).tolist()
                raise ValueError(
                    f"Malformed Raven Pro selection table {filename}: Channel values must be positive integers on rows {rows}."
                )
            channels = raven_channels.astype(int) - 1
        else:
            channels = np.full(len(df), -1, dtype=int)

        events_df = pd.DataFrame(
            {
                "name": labels,
                "start_seconds": begin_seconds,
                "stop_seconds": end_seconds,
                "channel": channels,
            }
        )
        if "Selection" in df.columns:
            selection_ids = self._numeric_column(df, "Selection", filename)
            invalid_selections = (selection_ids < 1) | (selection_ids != np.floor(selection_ids))
            if invalid_selections.any():
                rows = (np.flatnonzero(invalid_selections) + 2).tolist()
                raise ValueError(
                    f"Malformed Raven Pro selection table {filename}: Selection values must be positive integers on rows {rows}."
                )
            events_df["Selection"] = selection_ids.astype(int)
            events_df = events_df.drop_duplicates(subset=["Selection", "name", "start_seconds", "stop_seconds", "channel"])

        event_seconds = annot.Events.from_df(events_df)
        return event_seconds, event_seconds.categories

    @staticmethod
    def _numeric_column(df: pd.DataFrame, column: str, filename: str) -> np.ndarray:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        invalid = ~np.isfinite(values)
        if invalid.any():
            rows = (np.flatnonzero(invalid) + 2).tolist()
            raise ValueError(
                f"Malformed Raven Pro selection table {filename}: {column!r} must contain finite numeric values on rows {rows}."
            )
        return values


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
