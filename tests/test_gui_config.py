from pathlib import Path

import numpy as np
import pytest
import yaml

from xarray_behave.gui import gui_config


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_local_config_path_treats_zarr_as_file_like(tmp_path):
    folder = tmp_path / "recording"
    folder.mkdir()
    zarr = tmp_path / "recording.zarr"
    zarr.mkdir()

    assert gui_config.local_config_path(folder) == folder / ".das.yaml"
    assert gui_config.local_config_path(tmp_path / "audio.wav") == tmp_path / ".das.yaml"
    assert gui_config.local_config_path(zarr) == tmp_path / ".das.yaml"


def test_manager_merges_global_local_and_explicit_configs(tmp_path):
    home = tmp_path / "home"
    source = tmp_path / "data"
    source.mkdir()
    explicit = tmp_path / "profile.yaml"
    _write(
        home / ".das.yaml",
        {"version": 1, "viewer": {"spectrogram": {"colormap": "magma", "compression": 2}}},
    )
    _write(
        source / ".das.yaml",
        {"version": 1, "viewer": {"spectrogram": {"compression": 4}, "audio": {"waveform_all": False}}},
    )
    _write(explicit, {"version": 1, "viewer": {"spectrogram": {"colormap": "gray"}}})

    manager = gui_config.GuiConfigManager(str(explicit), home=home)
    config = manager.load_for_source(str(source))

    assert config["viewer"]["spectrogram"] == {"colormap": "gray", "compression": 4}
    assert config["viewer"]["audio"]["waveform_all"] is False


def test_manager_merges_project_between_global_and_explicit(tmp_path):
    home = tmp_path / "home"
    explicit = tmp_path / "profile.yaml"
    _write(home / ".das.yaml", {"version": 1, "viewer": {"spectrogram": {"colormap": "magma"}}})
    _write(explicit, {"version": 1, "viewer": {"spectrogram": {"colormap": "gray"}}})
    manager = gui_config.GuiConfigManager(str(explicit), home=home)

    config = manager.load_for_project(
        str(tmp_path / "songs.xbp.yaml"),
        {"version": 1, "viewer": {"spectrogram": {"colormap": "viridis", "compression": 2}}},
    )

    assert config["viewer"]["spectrogram"] == {"colormap": "gray", "compression": 2}


def test_source_specific_dialog_fields_are_never_saved():
    data = {
        "samplerate": 10_000,
        "data_set": "audio",
        "annotation_path": "/tmp/annotations.csv",
        "pixel_size_mm": 0.1,
        "target_samplingrate": 1_000,
        "filter_song": "yes",
        "f_low": 50,
    }

    assert gui_config.filter_dialog_data("from_file", data) == {
        "target_samplingrate": 1_000,
        "filter_song": "yes",
        "f_low": 50,
    }


def test_config_round_trip_filters_unknown_keys_and_omits_event_types(tmp_path, caplog):
    path = tmp_path / "profile.yaml"
    gui_config.write_config(
        path,
        {
            "version": 1,
            "unknown": True,
            "viewer": {"waveform": {"color": "#ffffff", "unknown": 1}},
            "event_types": [
                {"name": "pulse", "fixed_duration": True, "duration_seconds": 0.01},
                {"name": "song", "fixed_duration": False},
            ],
        },
    )

    loaded = gui_config.read_config(path)
    assert list(loaded) == ["version", "viewer"]
    assert loaded["viewer"]["waveform"] == {"color": "#ffffff"}
    assert "event_types" not in loaded
    assert "unknown GUI config" in caplog.text


def test_invalid_automatic_config_is_ignored_but_invalid_explicit_config_fails(tmp_path, caplog):
    home = tmp_path / "home"
    _write(home / ".das.yaml", {"version": 99})
    automatic = gui_config.GuiConfigManager(home=home)
    assert automatic.load_for_source() == {"version": 1}
    assert "Ignoring invalid automatic GUI config" in caplog.text

    explicit_path = tmp_path / "explicit.yaml"
    _write(explicit_path, {"version": 99})
    explicit = gui_config.GuiConfigManager(str(explicit_path), home=tmp_path / "empty-home")
    with pytest.raises(gui_config.ConfigError, match="version must be 1"):
        explicit.load_for_source()


def test_invalid_known_value_is_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"
    _write(path, {"version": 1, "viewer": {"spectrogram": {"levels": "invalid"}}})

    with pytest.raises(gui_config.ConfigError, match="levels must contain two"):
        gui_config.read_config(path)


def test_read_config_ignores_legacy_selection(tmp_path):
    path = tmp_path / ".das.yaml"
    _write(path, {"version": 1, "selection": {"event_type": "pulse", "audio_channel": "Channel 1"}})

    assert "selection" not in gui_config.read_config(path)


def test_write_config_omits_selection_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / ".das.yaml"
    gui_config.write_config(path, {"version": 1, "selection": {"event_type": "pulse"}})

    assert "selection" not in yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(tmp_path.glob(".*.tmp")) == []


def test_write_config_omits_event_types(tmp_path):
    path = tmp_path / ".das.yaml"
    gui_config.write_config(
        path,
        {
            "version": np.int64(1),
            "event_types": [
                {
                    "name": np.str_("sine"),
                    "fixed_duration": np.bool_(False),
                    "duration_seconds": np.float64(0.25),
                }
            ],
        },
    )

    loaded = gui_config.read_config(path)
    assert "event_types" not in loaded
