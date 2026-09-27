import numpy as np
import xarray as xr

from xarray_behave import event_utils


def test_detect_events_returns_start_stop_rows_for_points_and_intervals():
    ds = xr.Dataset(
        {
            "song_events": xr.DataArray(
                np.array(
                    [
                        [0, 0],
                        [1, 0],
                        [0, 1],
                        [0, 1],
                        [0, 0],
                    ]
                ),
                dims=["time", "event_types"],
                coords={
                    "time": [0.0, 0.1, 0.2, 0.3, 0.4],
                    "event_types": ["pulse", "song"],
                    "event_categories": ("event_types", ["event", "event"]),
                },
            )
        }
    )

    events = event_utils.detect_events(ds)

    np.testing.assert_allclose(events["pulse"], [[0.1, 0.1]])
    np.testing.assert_allclose(events["song"], [[0.2, 0.3]])


def test_eventtimes_to_traces_writes_points_and_intervals_as_events():
    ds = xr.Dataset(
        {
            "song_events": xr.DataArray(
                np.zeros((5, 2), dtype=int),
                dims=["time", "event_types"],
                coords={
                    "time": [0.0, 0.1, 0.2, 0.3, 0.4],
                    "event_types": ["pulse", "song"],
                    "event_categories": ("event_types", ["event", "event"]),
                },
            )
        }
    )

    updated = event_utils.eventtimes_to_traces(
        ds,
        {"pulse": np.array([[0.1, 0.1]]), "song": np.array([[0.2, 0.3]])},
    )

    np.testing.assert_array_equal(updated.song_events.sel(event_types="pulse").values, [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(updated.song_events.sel(event_types="song").values, [0, 0, 1, 1, 0])
    assert updated.event_categories.values.tolist() == ["event", "event"]


def test_update_traces_can_create_song_events_from_empty_dataset():
    ds = xr.Dataset(coords={"time": [0.0, 0.1, 0.2, 0.3]}, attrs={"target_sampling_rate_Hz": 10})

    updated = event_utils.update_traces(ds, {"song": np.array([[0.1, 0.2]])})

    assert updated.event_categories.values.tolist() == ["event"]
    np.testing.assert_array_equal(updated.song_events.sel(event_types="song").values, [0, 1, 1, 0])
