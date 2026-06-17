"""[summary]

TODO: From/to traces
"""

import numpy as np
import xarray as xr
import pandas as pd
from collections import UserDict
from typing import Optional, List, Dict, Tuple, Union


class Events(UserDict):
    def __init__(
        self,
        data: Optional[Dict[str, List[float]]] = None,
        categories: Optional[Dict[str, str]] = None,
        add_names_from_categories: bool = True,
    ):
        """[summary]

        Args:
            data: dict or Events
            categories (dict[str: str]): legacy name/category mapping. Values are normalized to "event".

        """
        if data is None:
            data = dict()

        super().__init__(data)

        for key, val in self.items():
            val = np.array(val)
            if val.ndim == 1:
                val = val[:, np.newaxis]
            if val.shape[1] == 1:
                val = np.concatenate((val, val), axis=1)
            if val.shape[1] == 2:
                channels = np.full((val.shape[0], 1), fill_value=-1)
                val = np.concatenate((val, channels), axis=1)

            self.data[key] = val
        self.categories = self._infer_categories()

        # drop nan
        self._drop_nan()

        # preserve names from input, but normalize categories. Event duration is
        # represented by start/stop values; category labels are legacy only.
        if hasattr(data, "categories"):
            for name, _cat in data.categories.items():
                if name in self:  # update only existing keys
                    self.categories[name] = "event"

        # update names from arg, but normalize categories.
        if categories is not None:
            for name, _cat in categories.items():
                if name in self:  # update only existing keys
                    self.categories[name] = "event"
                elif add_names_from_categories:
                    self.add_name(name=name, category="event")

    @classmethod
    def from_df(cls, df: pd.DataFrame, possible_event_names: Optional[List[str]] = None):
        if possible_event_names is None:
            possible_event_names = []
        if "channel" in df:
            channels = list(df["channel"])
        else:
            channels = None

        return cls.from_lists(
            df.name.values,
            df.start_seconds.values.astype(float),
            df.stop_seconds.values.astype(float),
            possible_event_names,
            channels=channels,
        )

    @classmethod
    def from_lists(
        cls,
        names: List[str],
        start_seconds: List[float],
        stop_seconds: List[float],
        possible_event_names: Optional[List[str]] = None,
        channels: Optional[List[int]] = None,
    ):
        if possible_event_names is None:
            possible_event_names = []
        unique_names = list(np.unique(names))  # use `np.unique` instead of `set` to have reproducible order of names for now
        unique_names.extend(possible_event_names)
        dct = {name: [] for name in unique_names}

        if channels is None:
            channels = np.full((len(start_seconds),), fill_value=-1)

        for name, start_second, stop_second, channel in zip(names, start_seconds, stop_seconds, channels):
            dct[name].append([start_second, stop_second, channel])

        return cls(dct)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset):
        start_seconds = np.array(ds.event_times.sel(event_time="start_seconds").data)
        stop_seconds = np.array(ds.event_times.sel(event_time="stop_seconds").data)
        if "channels" in ds.event_time:
            channels = np.array(ds.event_times.sel(event_time="channels").data)
        else:
            channels = None

        names = np.array(ds.event_names.data)
        if "possible_event_names" in ds.attrs:
            possible_event_names = ds.attrs["possible_event_names"]
        elif "possible_event_names" in ds.event_names.attrs:
            possible_event_names = ds.event_names.attrs["possible_event_names"]
        else:
            possible_event_names = []

        out = cls.from_lists(names, start_seconds, stop_seconds, possible_event_names, channels)
        if "event_categories" in ds:
            cats = {str(cat.event_types.data): str(cat.event_categories.data) for cat in ds.event_categories}
            out = cls(out, categories=cats)
        return out

    @classmethod
    def from_dict(cls, dct: Dict):
        """
        Args:
            dct (Dict[str, np.ndarray]): keys are event types, values is 2D array with
                                         either two columns containing start and stop seconds
                                         or three columns with start, stop, and channel.
        """
        names = []
        start_seconds = []
        stop_seconds = []
        if len(dct.values()):
            # check if there are annotations and if so, if they comes with channel information
            if list(dct.values())[0].ndim > 1 and list(dct.values())[0].shape[1] > 2:
                channels = []
            else:
                channels = None

            for k, v in dct.items():
                if len(v):  # check if there are annotations
                    names.extend([k] * v.shape[0])
                    start_seconds.extend(v[:, 0])
                    stop_seconds.extend(v[:, 1])
                    if channels is not None:
                        channels.extend(v[:, 2])
        out = cls.from_lists(names, start_seconds, stop_seconds, channels=channels)
        return out

    def update(self, new_dict: Dict):
        """Add all items in new_dict to self, overwrite existing items.
        Same as python's dict.update but also keeps track of categories.

        Args:
            new_dict ([type]): [description]
        """
        super().update(new_dict)
        if hasattr(self, "categories") and hasattr(new_dict, "categories"):
            self.categories.update(new_dict.categories)

    def _to_columns(self, preserve_empty: bool = True):
        names = []
        starts = []
        stops = []
        channels = []
        for name in self.names:
            values = np.asarray(self[name])
            if len(values):
                names.extend([name] * len(values))
                starts.append(values[:, 0])
                stops.append(values[:, 1])
                channels.append(values[:, 2])
            elif preserve_empty:
                names.append(name)
                starts.append(np.array([np.nan]))
                stops.append(np.array([np.nan]))
                channels.append(np.array([-1]))

        if starts:
            start_seconds = np.concatenate(starts).astype(float, copy=False)
            stop_seconds = np.concatenate(stops).astype(float, copy=False)
            channels = np.concatenate(channels).astype(float, copy=False)
        else:
            start_seconds = np.array([], dtype=float)
            stop_seconds = np.array([], dtype=float)
            channels = np.array([], dtype=float)
        return np.asarray(names), start_seconds, stop_seconds, channels

    def to_df(self, preserve_empty: bool = True, with_channels: bool = True):
        """Convert to pandas.DataFeame

        Args:
            preserve_empty (bool, optional):
                Preserve event names without annotations as rows with np.nan
                start and stop values. Defaults to True.
            with_channels (bool, optional):
                Will add channel information as 4th column to df.
                Defaults to True.

        Returns:
            pandas.DataFrame: with columns name, start_seconds, stop_seconds, channels (if with_channels). One row per event.
        """
        names, start_seconds, stop_seconds, channels = self._to_columns(preserve_empty=preserve_empty)
        data = {
            "name": names,
            "start_seconds": start_seconds,
            "stop_seconds": stop_seconds,
            "channel": channels,
        }
        df = pd.DataFrame(data)
        if not with_channels:
            del df["channel"]
        return df

    def to_lists(self, preserve_empty: bool = True):
        """[summary]

        Args:
            preserve_empty (bool, optional):
                Preserve event names without annotations as rows with np.nan
                start and stop values. Defaults to True.

        Returns:
            Tuple[List[str], List[float], List[float]: with names, start_seconds, stop_seconds.
        """
        return self._to_columns(preserve_empty=preserve_empty)

    def to_dataset(self):
        names, start_seconds, stop_seconds, channels = self.to_lists()

        da_names = xr.DataArray(name="event_names", data=np.array(names, dtype="U128"), dims=["index"])
        # da_channels = xr.DataArray(name="event_channels", data=np.array(names, dtype="U128"), dims=["index"])
        da_times = xr.DataArray(
            name="event_times",
            data=np.array([start_seconds, stop_seconds, channels]).T,
            dims=["index", "event_time"],
            coords={"event_time": ["start_seconds", "stop_seconds", "channels"]},
        )

        ds = xr.Dataset({da.name: da for da in [da_names, da_times]})
        ds.attrs["time_units"] = "seconds"
        ds.attrs["possible_event_names"] = self.names  # ensure that we preserve even names w/o events that get lost in to_df
        return ds

    def add_name(
        self,
        name: str,
        category: str = "event",
        times: Optional[np.array] = None,
        overwrite: bool = False,
        append: bool = False,
        sort_after_append: bool = False,
    ):
        """[summary]

        Args:
            name (str): Event name.
            category (str, optional): Legacy category label. Ignored; all names are events.
            times (np.array, optional): [N,2] array of floats with start (index 0) and end (index 1) of the annotations.
                                        Defaults to None.
            overwrite (bool, optional): Replace times and category if name exists. Defaults to False.
            append (bool, optional): Append times if name exists. Defaults to False.
            sort_after_append (bool, optional): Sort times by start_seconds. Defaults to False.
        """
        if times is None:
            times = np.zeros((0, 3))

        if name not in self or (name in self and overwrite):
            self.update({name: times})
            self.categories[name] = "event"
        elif name in self and append:
            self[name] = np.append(self[name], times, axis=0)
            if sort_after_append:
                self[name].sort(axis=0)

    def delete_name(self, name: str):
        """Delete all annotations with that name."""
        if name in self:
            del self[name]
        if name in self.categories:
            del self.categories[name]

    def _get_name_of_nearest(self, time: float, min_time: Optional[float] = None, max_time: Optional[float] = None):
        nearest_starts = dict()
        nearest_stops = dict()
        for name in self.keys():
            within_range_indices = self.select_range(name, min_time, max_time, strict=False)
            if len(within_range_indices):
                nearest_starts[name] = self._find_nearest(self.start_seconds(name)[within_range_indices], time)
                nearest_stops[name] = self._find_nearest(self.stop_seconds(name)[within_range_indices], time)

        if len(nearest_starts):
            nearest_starts_times = list(nearest_starts.values())
            nearest_stops_times = list(nearest_stops.values())
            nearest_names = list(nearest_starts.keys())
            distance_start = np.abs(time - np.array(nearest_starts_times))
            distance_stop = np.abs(time - np.array(nearest_stops_times))
            if np.min(distance_start) < np.min(distance_stop):
                name = nearest_names[np.nanargmin(distance_start)]
            else:
                name = nearest_names[np.nanargmin(distance_stop)]
        else:
            name = None
        return name

    def _get_index_of_nearest(
        self, time: float, name: str, tol: float = 0, min_time: Optional[float] = None, max_time: Optional[float] = None
    ):
        within_range_indices = self.select_range(name, min_time, max_time, strict=False)
        if len(within_range_indices):
            nearest_start = self._find_nearest(self.start_seconds(name)[within_range_indices], time)
            nearest_stop = self._find_nearest(self.stop_seconds(name)[within_range_indices], time)
        else:
            nearest_start = None
            nearest_stop = None

        if nearest_start is None and nearest_stop is None:
            return None, []

        if np.abs(nearest_start - time) < np.abs(nearest_stop - time):
            index = np.where(self.start_seconds(name) == nearest_start)[0][0]
            nearest_is_start = True
        else:
            index = np.where(self.stop_seconds(name) == nearest_stop)[0][0]
            nearest_is_start = False

        start = self.start_seconds(name)[index]
        stop = self.stop_seconds(name)[index]
        if not np.isfinite(start) or not np.isfinite(stop):
            event_at_time = False
        elif start == stop:
            event_at_time = min(np.abs(time - start), np.abs(time - stop)) <= tol
        else:
            lo, hi = sorted([start, stop])
            event_at_time = (lo <= time <= hi) or min(np.abs(time - lo), np.abs(time - hi)) <= tol

        if not event_at_time:
            return None
        else:
            return index

    def change_name(
        self,
        time: float,
        new_name: str,
        tol: float = 0,
        min_time: Optional[float] = None,
        max_time: Optional[float] = None,
        old_name: Optional[str] = None,
    ) -> Tuple[Optional[List[int]], Optional[str], Optional[str]]:
        """Change the name of the annotation.

        Args:
            time (float): Time point of/near annotation.
            new_name (str): New name for the annotation
            tol (float, optional): Tolerance for matching events. Defaults to 0.
            min_time (Optional[float], optional): _description_. Defaults to None.
            max_time (Optional[float], optional): _description_. Defaults to None.
            old_name (Optional[str]): name of event to move. Defaults to None.
        Returns:
            Tuple[List[int], str, str]: ([start_seconds, stop_seconds], old_name, new_name
            Tuple[None, None, None] if no event near time, or new_name is old_name
        """
        if old_name is None:
            name = self._get_name_of_nearest(time, min_time, max_time)
        else:
            name = old_name

        # nothing to do
        if name is None or name == new_name:
            return None, None, None

        index = self._get_index_of_nearest(time, name, tol, min_time, max_time)
        if index is not None:
            changed_time = self[name][index, :]
            old_name = name
            self[old_name] = np.delete(self[old_name], index, axis=0)
            self.add_time(new_name, *changed_time)
            return changed_time, old_name, new_name
        else:
            return None, None, None

    def add_time(
        self,
        name: str,
        start_seconds: float,
        stop_seconds: float = None,
        add_new_name: bool = True,
        category: Optional[str] = None,
        channel: int = -1,
    ):
        """Add a new event.

        Args:
            name (str): Event name.
            start_seconds (float): Event start.
            stop_seconds (float, optional): Event stop. Defaults to None (use start_seconds).
            add_new_name (bool, optional): Add new event name if name does not exist yet. Defaults to True.
            category (str, optional): Legacy category label. Ignored.
            channel (int, optional): Index into channel.
        """
        if stop_seconds is None:
            stop_seconds = start_seconds

        if name not in self and add_new_name:
            if category is None:
                category = "event"
            self.add_name(name, category=category)

        data = sorted([start_seconds, stop_seconds])  # sort to make sure start is before stop
        data.append(channel)

        self[name] = np.insert(self[name], len(self[name]), data, axis=0)  # why not use append?

    def move_time(self, name: str, old_time: Union[float, Tuple], new_time: Union[float, Tuple]):
        """[summary]

        Args:
            name ([type]): [description]
            old_time ([type]): [description]
            new_time ([type]): [description]
        """
        if isinstance(old_time, float):
            old_time = [old_time, old_time]
        if isinstance(new_time, float):
            new_time = [new_time, new_time]

        hits = np.all(self[name][:, : len(old_time)] == old_time, axis=1)
        self[name][hits, : len(new_time)] = new_time

    def delete_time(
        self,
        time: float,
        name: Optional[str] = None,
        tol: float = 0,
        min_time: Optional[float] = None,
        max_time: Optional[float] = None,
    ):
        """[summary]

        Args:
            name ([type], optional): [description]. Defaults to None.
            time ([type], optional): [description]. Defaults to None.
            tol (int, optional): [description]. Defaults to 0.
            min_time ([type], optional): [description]. Defaults to None.
            max_time ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if name is None:
            name = self._get_name_of_nearest(time, min_time, max_time)
            if name is None:
                return None, []

        index = self._get_index_of_nearest(time, name, tol, min_time, max_time)
        if index is not None:
            deleted_time = self[name][index, :]
            deleted_name = name
            self[name] = np.delete(self[name], index, axis=0)
        else:
            deleted_time = []
            deleted_name = None

        return deleted_name, deleted_time

    def select_range(self, name: str, t0: Optional[float] = None, t1: Optional[float] = None, strict: bool = True):
        """Get indices of events within the range.

        Need to start and stop after t0 and before t1 (non-inclusive bounds).

        Args:
            name (str): [description]
            t0 (float, optional): [description]
            t1 (float, optional): [description]
            strict (bool, optional): if true, only matches events that start AND stop within the range,
                                     if false, matches events that start OR stop within the range

        Returns:
            List[uint]: List of indices of events within the range
        """
        if t0 is None:
            t0 = 0
        if t1 is None:
            t1 = np.inf

        if strict:
            within_range = np.logical_and(self.start_seconds(name) > t0, self.stop_seconds(name) < t1)
        else:
            starts_in_range = np.logical_and(self.start_seconds(name) > t0, self.start_seconds(name) < t1)
            stops_in_range = np.logical_and(self.stop_seconds(name) > t0, self.stop_seconds(name) < t1)
            within_range = np.logical_or(starts_in_range, stops_in_range)
        within_range_indices = np.where(within_range)[0]
        return within_range_indices

    def filter_range(self, name: str, t0: float, t1: float, strict: bool = False):
        """Returns events within the range.

        Need to start and stop after t0 and before t1 (non-inclusive bounds).

        Args:
            name ([type]): [description]
            t0 ([type]): [description]
            t1 ([type]): [description]
            strict (bool): if true, only matches events that start AND stop within the range,
                           if false, matches events that start OR stop within the range
        Returns:
            List[float]: [N, 2] list of start_seconds and stop_seconds in the range
        """
        indices = self.select_range(name, t0, t1, strict)
        return self[name][indices, :]

    def delete_range(self, name: str, t0: float, t1: float, strict: bool = True):
        """Deletes events within the range.

        Need to start and stop after t0 and before t1 (non-inclusive bounds).

        Args:
            name ([type]): [description]
            t0 ([type]): [description]
            t1 ([type]): [description]
            strict (bool): if true, only matches events that start AND stop within the range,
                           if false, matches events that start OR stop within the range
        Returns:
            int: number of deleted events
        """
        indices = self.select_range(name, t0, t1, strict)
        self[name] = np.delete(self[name], indices, axis=0)
        return indices

    def sort(self, names: Optional[List[str]] = None):
        """Sort annotations by start time.

        Args:
            names (Optional[List[str]], optional): _description_. Defaults to None.
        """
        if names is None:
            names = self.names

        for name in names:
            self[name] = self[name][np.argsort(self[name][:, 0]), :]

    def find_next(self, t: float, names: Optional[List[str]] = None):
        """Find event starting after `t` of type in names.

        Args:
            t (float): Current time in seconds.
            names (Optional[List[str]], optional):
                List of event names to find next time in.
                Defaults to None (search in all events of any name).

        Returns:
            float or None: time of nearest next event. None if there is no next.
        """
        if names is None:
            names = self.names

        nxt = []

        for name in names:
            self.sort([name])
            cmp = self[name][:, 0] > t
            if np.any(cmp):
                nxt.append(self[name][np.argmax(cmp), 0])
        if len(nxt):
            return np.min(nxt)

    def find_prev(self, t: float, names: Optional[List[str]] = None):
        """Find event ending before `t` of type in names

        Args:
            t (float): Current time in seconds.
            names (Optional[List[str]], optional):
                List of event names to find prev time in.
                Defaults to None (search in all events of any name).

        Returns:
            float or None: time of nearest previous event. None if there is no previous.
        """
        if names is None:
            names = self.names

        nxt = []
        for name in names:
            self.sort([name])
            cmp = self[name][:, 1] < t
            if np.any(cmp):
                nxt.append(self[name][np.argmin(cmp) - 1, 0])
        if len(nxt):
            return np.max(nxt)  # * self.fs_song

    def _find_nearest(self, array: np.array, value: float):
        if not len(array):
            return None
        else:
            idx = (np.abs(array - value)).argmin()
            return array[idx]

    def _infer_categories(self):
        return {name: "event" for name in self.names}

    def _drop_nan(self):
        # remove entries with nan stop or start (but keep their name)
        for name in self.names:
            nan_events = np.logical_or(np.isnan(self.start_seconds(name)), np.isnan(self.stop_seconds(name)))
            self[name] = self[name][~nan_events]

    @property
    def names(self):
        return list(self.keys())

    def start_seconds(self, key: str):
        return self[key][:, 0]

    def stop_seconds(self, key: str):
        return self[key][:, 1]

    def channels(self, key: str):
        return self[key][:, 2]

    def duration_seconds(self, key: str):
        return self[key][:, 1] - self[key][:, 0]
