"""Create self-documenting xarray dataset from behavioral recordings and annotations."""

import numpy as np
from .io.samplestamps import SampStamp, SimpleStamp
import scipy.interpolate
import scipy.stats
import xarray as xr
import zarr
import logging
import os.path
import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from . import loaders as ld, metrics as mt, event_utils, annot, io

logger = logging.getLogger(__name__)


def assemble(
    datename: str = "",
    root: str = "",
    dat_path: str = "dat",
    res_path: str = "res",
    filepath_timestamps: Optional[Union[Path, str]] = None,
    filepath_video: Optional[Union[Path, str]] = None,
    filepath_timestamps_ball: Optional[Union[Path, str]] = None,
    filepath_daq: Optional[Union[Path, str]] = None,
    filepath_annotations: Optional[Union[Path, str]] = None,
    filepath_definitions: Optional[Union[Path, str]] = None,
    filepath_tracks: Optional[Union[Path, str]] = None,
    filepath_poses: Optional[Union[Path, str]] = None,
    target_sampling_rate: float = 1_000,
    audio_sampling_rate: Optional[float] = None,
    audio_channels: Optional[List[int]] = None,
    audio_dataset: Optional[str] = None,
    event_names: Optional[List[str]] = None,
    event_categories: Optional[List[str]] = None,
    resample_video_data: bool = True,
    include_song: bool = True,
    include_tracks: bool = True,
    include_poses: bool = True,
    include_balltracker: bool = True,
    include_movieparams: bool = True,
    fix_fly_indices: bool = True,
    pixel_size_mm: Optional[float] = None,
    lazy_load_song: bool = False,
    make_song_events: bool = True,
) -> xr.Dataset:
    """[summary]

    Args:
        datename (str, optional): Used to infer file paths.
                           Constructed paths follow this schema f"{root}/{dat_path|res_path}/{datename}/{datename}_suffix.extension".
                           Can be overridden for some paths with arguments below. Defaults to None.
        root (str, optional): Used to infer file paths. Defaults to ''.
        dat_path (str, optional): Used to infer paths of video, daq, and timestamps files. Defaults to 'dat'.
        res_path (str, optional): Used to infer paths of annotation, tracking etc files. Defaults to 'res'.
        filepath_timestamps (str, optional): Path to the timestamps file. Defaults to None (infer path from `datename` and `dat_path`).
        filepath_timestamps_ball (str, optional): Path to the timestamps file. Defaults to None (infer path from `datename` and `dat_path`).
        filepath_video (str, optional): Path to the video file. Defaults to None (infer path from `datename` and `dat_path`).
        filepath_daq (str, optional): Path to the daq file. Defaults to None (infer path from `datename` and `dat_path`).
        filepath_annotations (str, optional): Path to annotations file. Defaults to None (infer path from `datename` and `res_path`).
        filepath_tracks (str, optional): Path to tracks file. Defaults to None (infer path from `datename` and `res_path`).
        filepath_poses (str, optional): Path to poses file. Defaults to None (infer path from `datename` and `res_path`).
        target_sampling_rate (int, optional): Sampling rate in Hz for pose and annotation data. Defaults to 1000.
        audio_sampling_rate (int, optional): Audio sampling rate in Hz.
                                             Used to override sampling rate inferred from file or to provide sampling rate if missing from file.
                                             Defaults to None (use inferred sampling rate).
        audio_channels (List[int], optional): Defaults to None (all channels).
        audio_dataset (str, optional): Name of the dataset in NPZ and H5 files that contains the audio data. Defaults to 'data'.
        event_names (List[str], optional): List of event names to initialize dataset with. Defaults to [].
        event_categories (List[str], optional): 'segment' or 'event' for each item in `event_names`. Defaults to 'segment'.
        resample_video_data (bool, optional): Or keep video with original frame times. Defaults to True.
        include_song (bool, optional): [description]. Defaults to True.
        include_tracks (bool, optional): [description]. Defaults to True.
        include_poses (bool, optional): [description]. Defaults to True.
        include_balltracker (bool, optional): [description]. Defaults to True.
        include_movieparams (bool, optional): [description]. Defaults to True.
        fix_fly_indices (bool, optional): Will attempt to load swap info and fix fly id's accordingly, Defaults to True.
        pixel_size_mm (float, optional): Size of a pixel (in mm) in the video. Used to convert tracking data to mm.
        lazy_load_song (float): Memmap data via dask. If false, full array will be loaded into memory. Defaults to False
        make_song_events (bool, optional): Make binary matrix of song events. Defaults to True.
    Returns:
        xarray.Dataset
    """
    song_raw = None
    non_song_raw = None

    # load RECORDING and TIMING INFORMATION
    if filepath_daq is None:
        filepath_daq = Path(root, dat_path, datename, f"{datename}_daq.h5")
        filepath_daq_is_custom = False
    else:
        filepath_daq_is_custom = True

    if filepath_video is None:
        filepath_video = str(Path(root, dat_path, datename, f"{datename}.mp4"))
        if not os.path.exists(filepath_video):  # try avi
            filepath_video = str(Path(root, dat_path, datename, f"{datename}.avi"))

    if filepath_timestamps is None:  # for video
        basename = os.path.splitext(filepath_video)[0]
        filepath_timestamps = f"{basename}_timestamps.h5"
        if not os.path.exists(filepath_timestamps):  # try avi
            filepath_timestamps = Path(root, dat_path, datename, f"{datename}_timestamps.h5")

    # Create samplestamps object
    ss = None
    if os.path.exists(filepath_daq) and os.path.exists(filepath_timestamps):
        ss, last_sample_number, sampling_rate = ld.load_times(filepath_timestamps, filepath_daq)
        if sampling_rate is None:
            sampling_rate = audio_sampling_rate
        path_tried = (filepath_daq, filepath_timestamps)
    elif os.path.exists(filepath_video):  # Video (+tracks) w/o DAQ
        # if there is only the video, generate fake samples from fps
        from videoreader import VideoReader

        vr = VideoReader(filepath_video)

        if os.path.exists(filepath_timestamps):
            _, frame_times = io.timestamps.CamStamps().load(filepath_timestamps)
            frame_times -= frame_times[0]
        else:
            frame_times = np.arange(0, vr.number_of_frames, 1) / vr.frame_rate

        if target_sampling_rate == 0 or target_sampling_rate is None:
            resample_video_data = False
            target_sampling_rate = vr.frame_rate

        frame_times[-1] = frame_times[-2]  # for auto-monotonize to not mess everything up

        sampling_rate = 10 * target_sampling_rate
        sample_times = np.arange(frame_times[0], frame_times[-1], 1)  # first to last frame, in 1s steps
        sample_numbers = np.round(sample_times * sampling_rate)
        last_sample_number = sample_numbers[-1]

        ss = SampStamp(sample_times, frame_times, sample_numbers=sample_numbers)
        path_tried = (filepath_video,)
    elif os.path.exists(filepath_daq) and not os.path.exists(filepath_timestamps):  # Audio (+ annotations) only
        # if there is no video and no timestamps - generate fake from samplerate and number of samples
        # THIS SHOULD BE THE FIRST THING WE DO:
        logger.info("Loading audio data:")
        if not filepath_daq_is_custom:
            basename = os.path.join(root, dat_path, datename, datename)
        else:
            basename = filepath_daq

        audio_loader = io.get_loader(
            kind="audio",
            basename=basename,
            basename_is_full_name=filepath_daq_is_custom,
        )
        if not audio_loader and filepath_daq_is_custom:
            audio_loader = io.audio.AudioFile(basename)

        try:
            song_raw, non_song_raw, sampling_rate = audio_loader.load(
                audio_loader.path,
                song_channels=audio_channels,
                return_nonsong_channels=True,
                lazy=lazy_load_song,
                audio_dataset=audio_dataset,
            )
            logger.info(f"   {audio_loader.path} loaded using {audio_loader.NAME}.")
            if sampling_rate is None:
                sampling_rate = audio_sampling_rate
            last_sample_number = len(song_raw)
            ss = SimpleStamp(sampling_rate)
        except:
            raise ValueError(f"Loading {audio_loader.path} using {audio_loader.NAME} failed.")
        path_tried = (filepath_daq,)
    else:
        raise ValueError(f"Nothing found at {(filepath_daq, filepath_video, filepath_timestamps)}.")

    if ss is None:
        raise ValueError(f"No data found at {path_tried}.")

    if filepath_timestamps_ball is None:
        filepath_timestamps_ball = Path(root, dat_path, datename, f"{datename}_ball_timestamps.h5")
    if os.path.exists(filepath_daq) and os.path.exists(filepath_timestamps_ball):
        ss_ball, last_sample_number_ball, sampling_rate_ball = ld.load_times(filepath_timestamps_ball, filepath_daq)

    filepath_timestamps_movie = Path(root, res_path, datename, f"{datename}_movieframes.csv")
    ss_movie = None
    if os.path.exists(filepath_daq) and os.path.exists(filepath_timestamps_movie):
        ss_movie, last_sample_number_movie, sampling_rate_movie = ld.load_movietimes(filepath_timestamps_movie, filepath_daq)

    if target_sampling_rate == 0 or target_sampling_rate is None:
        resample_video_data = False
    fps = 1 / np.median(np.diff(ss.frames2times.y))
    if not resample_video_data:
        logger.info(f"  setting targetsamplingrate to avg. fps ({fps}).")
        target_sampling_rate = fps

    # # LOAD VIDEO
    # with_video = False
    # include_video = False  #True
    # if include_video:
    #     logger.info("Loading video:")
    #     if filepath_video is not None:
    #         video_loader = io.get_loader(kind="video", basename=filepath_video, basename_is_full_name=True)
    #     else:
    #         video_loader = io.get_loader(kind="video", basename=os.path.join(root, res_path, datename, datename))
    #     # breakpoint()
    #     if video_loader:
    #         try:
    #             xr_video = video_loader.make(video_loader.path)
    #             xr_video = add_time(xr_video, ss, dim="frame_number")
    #             xr_video = xr_video.drop_indexes(["frame_number"])
    #             xr_video = xr_video.set_xindex("frametimes")
    #             logger.info(f"  {video_loader.path} loaded.")
    #             with_video = True
    #         except Exception as e:
    #             logger.info(f"  Loading {video_loader.path} failed.")
    #             logger.exception(e)
    #     else:
    #         logger.info("   Found no tracks.")
    #     logger.info("Done.")

    # LOAD TRACKS
    with_tracks = False
    with_fixed_tracks = False
    if include_tracks:
        logger.info("Loading tracks:")
        if filepath_tracks is not None:
            tracks_loader = io.get_loader(kind="tracks", basename=filepath_tracks, basename_is_full_name=True)
        else:
            tracks_loader = io.get_loader(kind="tracks", basename=os.path.join(root, res_path, datename, datename))
        if tracks_loader:
            try:
                xr_tracks = tracks_loader.make(tracks_loader.path)
                xr_tracks = add_time(xr_tracks, ss, dim="frame_number")

                with_fixed_tracks = tracks_loader.path.endswith("_tracks_fixed.h5")
                logger.info(f"  {tracks_loader.path} loaded.")
                with_tracks = True
            except Exception as e:
                logger.info(f"  Loading {tracks_loader.path} failed.")
                logger.exception(e)
        else:
            logger.info("   Found no tracks.")
        logger.info("Done.")

    # LOAD POSES
    with_poses = False
    poses_from = None
    if include_poses:
        logger.info("Loading poses:")
        if filepath_poses is not None:
            poses_loader = io.get_loader(kind="poses", basename=filepath_poses, basename_is_full_name=True)
        else:
            poses_loader = io.get_loader(kind="poses", basename=os.path.join(root, res_path, datename, datename))
        if poses_loader:
            try:
                xr_poses, xr_poses_allo = poses_loader.make(poses_loader.path)
                xr_poses = add_time(xr_poses, ss, dim="frame_number")
                xr_poses_allo = add_time(xr_poses_allo, ss, dim="frame_number")

                with_poses = True
                poses_from = poses_loader.NAME
                logger.info(f"   {poses_loader.path} loaded.")
            except Exception as e:
                logger.info(f"   Loading {poses_loader.path} failed.")
                logger.exception(e)
        else:
            logger.info("   Found no poses.")
        logger.info("Done.")

    # LOAD BALLTRACKER
    with_balltracker = False
    if include_balltracker:
        logger.info("Loading ball tracker:")
        balltracker_loader = io.get_loader(kind="balltracks", basename=os.path.join(root, res_path, datename, datename))
        if balltracker_loader:
            try:
                xr_balltracks = balltracker_loader.make(balltracker_loader.path)
                xr_balltracks = add_time(xr_balltracks, ss_ball, dim="frame_number_ball", suffix="_ball")

                logger.info(f"   {balltracker_loader.path} loaded.")
                with_balltracker = True
            except Exception as e:
                logger.info(f"   Loading {balltracker_loader.path} failed.")
                logger.exception(e)
        else:
            logger.info("   Found no balltracker data.")
        logger.info("Done.")

    # LOAD MOVIEPARAMS
    with_movieparams = False
    if include_movieparams:
        logger.info("Loading movie params:")
        if ss_movie is None:
            logger.warning("   Failed loading movie params - no ss_movie")
        else:
            movieparams_loader = io.get_loader(
                kind="movieparams",
                basename=os.path.join(root, dat_path, datename, datename),
            )
            if movieparams_loader:
                try:
                    xr_movieparams = movieparams_loader.make(movieparams_loader.path)
                    xr_movieparams = add_time(
                        xr_movieparams,
                        ss_movie,
                        dim="frame_number_movie",
                        suffix="_movie",
                    )

                    logger.info(f"   {movieparams_loader.path} loaded.")
                    with_movieparams = True
                except Exception as e:
                    logger.info(f"   Loading {movieparams_loader.path} failed.")
                    logger.exception(e)
            else:
                logger.warning("   Found no movie params data.")
            logger.info("Done.")

    # Init empty and event data
    auto_event_seconds: Dict[str, Any] = {}
    auto_event_categories: Dict[str, Any] = {}

    if event_names is None:
        event_names = []
    if event_categories is None:
        event_categories = []

    if len(event_names) != len(event_categories):
        logger.warning(
            f"event_names and event_categories need to have same length - have {len(event_names)} and {len(event_categories)}."
        )
        event_categories = []

    if event_names and not event_categories:
        logger.info("No event_categories specified - defaulting to segments")
        event_categories = ["segment"] * len(event_names)
    manual_event_seconds: Dict[str, Any] = {name: np.zeros((0,)) for name in event_names}
    manual_event_categories: Dict[str, Any] = {nam: cat for nam, cat in zip(event_names, event_categories)}

    if include_song:
        logger.info("Loading automatic annotations:")
        custom_filepath_annotations = filepath_annotations is not None
        if not custom_filepath_annotations:
            filepath_annotations = os.path.join(root, res_path, datename, datename)

        annot_loader = io.get_loader(
            kind="annotations",
            basename=filepath_annotations,
            stop_after_match=False,
            basename_is_full_name=custom_filepath_annotations,
        )
        if annot_loader:
            if isinstance(annot_loader, str):
                annot_loader = list(annot_loader)

            for al in annot_loader:
                try:
                    this_event_seconds, this_event_categories = al.load()
                    auto_event_seconds.update(this_event_seconds)
                    auto_event_categories.update(this_event_categories)
                    logger.info(f"   {al.path} loaded.")
                except Exception as e:
                    logger.info(f"   Loading {al.path} failed.")
                    logger.exception(e)
        else:
            logger.info(f"   Found no automatic annotations.")
        logger.info("Done.")

        # load MANUAL SONG ANNOTATIONS
        logger.info("Loading manual annotations:")
        manual_annot_loader = io.get_loader(
            kind="annotations_manual",
            basename=filepath_annotations,
            basename_is_full_name=custom_filepath_annotations,
        )
        if manual_annot_loader:
            try:
                manual_event_seconds_loaded, manual_event_categories_loaded = manual_annot_loader.load(manual_annot_loader.path)
                manual_event_seconds.update(manual_event_seconds_loaded)
                manual_event_categories.update(manual_event_categories_loaded)
                logger.info(f"   {manual_annot_loader.path} loaded.")
            except Exception as e:
                logger.info(f"   Loading {manual_annot_loader.path} failed.")
                logger.exception(e)

        else:
            logger.info("   Found no manual automatic annotations.")
        logger.info("Done.")

        # load SONG DEFINITIONS
        logger.info("Loading song definitions:")
        custom_filepath_definitions = filepath_definitions is not None
        if not custom_filepath_definitions:
            filepath_definitions = os.path.join(
                root, res_path, datename, datename
            )  # or construct form audio file name - .ext + _definitions.csv

        definitions_loader = io.get_loader(
            kind="definitions_manual",
            basename=filepath_definitions,
            basename_is_full_name=custom_filepath_definitions,
        )
        if definitions_loader:
            try:
                manual_event_seconds_loaded, manual_event_categories_loaded = definitions_loader.load(definitions_loader.path)
                # add non-existing keys w/o overwriting values for existing ones
                manual_event_seconds.update(
                    {k: v for k, v in manual_event_seconds_loaded.items() if k not in manual_event_seconds}
                )
                manual_event_categories.update(manual_event_categories_loaded)
                logger.info(f"   {definitions_loader.path} loaded.")
            except Exception as e:
                logger.info(f"   Loading {definitions_loader.path} failed.")
                logger.exception(e)

        else:
            logger.info("   Found no song definitions.")
        logger.info("Done.")

        # load RAW song traces
        if song_raw is None:
            logger.info("Loading audio data:")
            if not filepath_daq_is_custom:
                basename = os.path.join(root, dat_path, datename, datename)
            else:
                basename = filepath_daq

            audio_loader = io.get_loader(
                kind="audio",
                basename=basename,
                basename_is_full_name=filepath_daq_is_custom,
            )
            if not audio_loader and filepath_daq_is_custom:
                audio_loader = io.audio.AudioFile(basename)

            if audio_loader:
                try:
                    song_raw, non_song_raw, samplerate = audio_loader.load(
                        audio_loader.path,
                        return_nonsong_channels=True,
                        lazy=lazy_load_song,
                    )
                    logger.info(f"   {audio_loader.path} with shape {song_raw.shape} loaded using {audio_loader.NAME}.")
                except Exception as e:
                    logger.info(f"   Loading {audio_loader.path} using {audio_loader.NAME} failed.")
                    logger.exception(e)
            logger.info("Done.")

        if song_raw is None:
            # song_raw = scipy.sparse.csr_array((int(sample_numbers[-1]), 1), dtype=bool)  # mem efficient but does not work with xr atm
            song_raw = np.zeros((int(sample_numbers[-1]), 1), dtype=bool)

    # merge manual and auto events -
    # manual will overwrite existing auto events with the same key
    event_seconds = auto_event_seconds.copy()
    event_seconds.update(manual_event_seconds)
    event_categories = auto_event_categories.copy()
    event_categories.update(manual_event_categories)

    event_seconds = ld.fix_keys(event_seconds)
    event_categories = ld.fix_keys(event_categories)

    # PREPARE sample/time/framenumber grids
    if not with_tracks:
        first_tracked_frame = int(ss.frame(0))
        last_tracked_frame = int(ss.frame(last_sample_number))
        logger.info(
            f"No tracks - setting first/last tracked frame numbers to those of the first/last sample in the recording ({first_tracked_frame}, {last_tracked_frame})."
        )
    else:
        first_tracked_frame, last_tracked_frame = (
            int(xr_tracks.frame_number[0]),
            int(xr_tracks.frame_number[-1]),
        )
        logger.info(f"Tracked frame {first_tracked_frame} to {last_tracked_frame}.")

    # construct desired sample grid for data
    step = sampling_rate / target_sampling_rate
    if resample_video_data:
        # in case the DAQ stopped before the video
        last_sample_with_frame = np.min((last_sample_number, ss.sample(frame=last_tracked_frame - 1))).astype(np.intp)
        target_samples = np.arange(0, last_sample_with_frame, step, dtype=np.uintp)
    else:
        frame_numbers = np.arange(first_tracked_frame, last_tracked_frame)
        frame_samples = ss.sample(frame_numbers)
        frame_samples = interp_duplicates(frame_samples)
        target_samples = frame_samples

    ref_time = ss.sample_time(0)  # first sample corresponds to 0 seconds
    time = ss.sample_time(target_samples) - ref_time

    # PREPARE DataArrays
    dataset_data = dict()
    logger.info("Making all datasets:")
    if song_raw is not None:
        if 0 not in song_raw.shape:  # xr fails saving zarr files with 0-size along any dim
            song_raw = xr.DataArray(
                data=song_raw[:, :],
                dims=["sampletime", "channels"],
                coords={
                    "sampletime": ss.sample_time(np.arange(song_raw.shape[0])) - ref_time,
                },
                attrs={
                    "description": "Raw song recording (multi channel).",
                    "sampling_rate_Hz": sampling_rate,
                    "time_units": "seconds",
                    "amplitude_units": "volts",
                },
            )
            dataset_data["song_raw"] = song_raw

    if non_song_raw is not None:
        if 0 not in non_song_raw.shape:  # xr fails saving zarr files with 0-size along any dim
            non_song_raw = xr.DataArray(
                data=non_song_raw[:, :],
                dims=["sampletime", "no_song_channels"],
                coords={
                    "sampletime": ss.sample_time(np.arange(non_song_raw.shape[0])) - ref_time,
                },
                attrs={
                    "description": "Non song (stimulus) data.",
                    "sampling_rate_Hz": sampling_rate,
                    "time_units": "seconds",
                    "amplitude_units": "volts",
                },
            )
            dataset_data["non_song_raw"] = non_song_raw

    logger.info("   Segmentations")
    if make_song_events:
        song_events = np.zeros((len(time), len(event_seconds)), dtype=np.int16)
        song_events = xr.DataArray(
            data=song_events,  # start with empty song_events matrix
            dims=["time", "event_types"],
            coords={
                "time": time,
                "event_types": list(event_seconds.keys()),
                "event_categories": (("event_types"), list(event_categories.values())),
            },
            attrs={
                "description": "Event times as boolean arrays.",
                "sampling_rate_Hz": sampling_rate / step,
                "time_units": "seconds",
                "event_times": event_seconds,
            },
        )
        # now populate from event_times attribute
        song_events_ds = event_utils.eventtimes_to_traces(
            song_events.to_dataset(name="song_events"), song_events.attrs["event_times"]
        )
        dataset_data["song_events"] = song_events_ds.song_events

    try:
        ds_eventtimes = annot.Events(event_seconds).to_dataset()
        dataset_data["event_times"] = ds_eventtimes.event_times
        dataset_data["event_names"] = ds_eventtimes.event_names
        dataset_data["event_names"].attrs["possible_event_names"] = ds_eventtimes.attrs["possible_event_names"]
    except Exception as e:
        logger.error("      Failed to generate event_times data arrays:")
        logger.exception(e)

    # BODY POSITION
    if pixel_size_mm is None:
        pixel_size_mm = np.nan

    # if with_video:
    #     logger.info("   Video")
    #     # set frametimes rel to ref-time
    #     xr_video["frametimes_rel"] = xr_video.frametimes - ref_time
    #     xr_video["frametimes"] = xr_video["frametimes_rel"]
    #     xr_video = xr_video.drop_vars("frametimes_rel")
    #     xr_video = xr_video.set_xindex("frametimes")
    #     dataset_data["video"] = xr_video

    if with_tracks:
        logger.info("   Tracking")
        xr_tracks = align_time(
            xr_tracks,
            ss,
            target_samples,
            ref_time=ref_time,
            target_time=time,
            extrapolate=True,
        )
        xr_tracks.attrs.update(
            {
                "description": 'coords are "allocentric" - rel. to the full frame',
                "sampling_rate_Hz": sampling_rate / step,
                "time_units": "seconds",
                "video_fps": fps,
                "spatial_units": "pixels",
                "pixel_size_mm": pixel_size_mm,
                "tracks_fixed": with_fixed_tracks,
            }
        )
        dataset_data["body_positions"] = xr_tracks

    # POSES
    if with_poses:
        logger.info("   Poses")
        xr_poses = align_time(xr_poses, ss, target_samples, ref_time=ref_time, target_time=time)
        xr_poses.attrs.update(
            {
                "description": 'coords are "egocentric" - rel. to box',
                "sampling_rate_Hz": sampling_rate / step,
                "time_units": "seconds",
                "video_fps": fps,
                "spatial_units": "pixels",
                "pixel_size_mm": pixel_size_mm,
                "poses_from": poses_from,
            }
        )
        dataset_data["pose_positions"] = xr_poses

        # poses in ALLOcentric (frame-relative) coordinates
        xr_poses_allo = align_time(xr_poses_allo, ss, target_samples, ref_time=ref_time, target_time=time)
        xr_poses_allo.attrs.update(
            {
                "description": 'coords are "allocentric" - rel. to frame',
                "sampling_rate_Hz": sampling_rate / step,
                "time_units": "seconds",
                "video_fps": fps,
                "spatial_units": "pixels",
                "pixel_size_mm": pixel_size_mm,
                "poses_from": poses_from,
            }
        )

        dataset_data["pose_positions_allo"] = xr_poses_allo

    # BALLTRACKS
    if with_balltracker:
        logger.info("   Balltracker")
        xr_balltracks = align_time(
            xr_balltracks,
            ss_ball,
            target_samples,
            target_time=time,
            dim="frame_number_ball",
            suffix="_ball",
            ref_time=ref_time,
        )
        xr_balltracks.attrs.update(
            {
                "description": "",
                "sampling_rate_Hz": sampling_rate / step,
                "time_units": "seconds",
                "video_fps": fps,
            }
        )
        dataset_data["balltracks"] = xr_balltracks

    # MOVIEPARAMS
    if with_movieparams:
        logger.info("   Movieparams")
        xr_movieparams = align_time(
            xr_movieparams,
            ss_movie,
            target_samples,
            dim="frame_number_movie",
            suffix="_movie",
            ref_time=ref_time,
            target_time=time,
            extrapolate=True,
        )
        xr_movieparams.attrs.update(
            {
                "description": "",
                "sampling_rate_Hz": sampling_rate / step,
                "time_units": "seconds",
                "video_fps": fps,
            }
        )
        dataset_data["movieparams"] = xr_movieparams

    # MAKE THE DATASET
    logger.info("   Assembling")

    dataset = xr.Dataset(dataset_data, attrs={})
    if "time" not in dataset:
        dataset.coords["time"] = time
    if "sampletime" not in dataset:
        dataset.coords["sampletime"] = time
    if "nearest_frame" not in dataset:
        dataset.coords["nearest_frame"] = (
            ("time"),
            (ss.times2frames(dataset["time"] + ref_time).astype(np.intp)),
        )

    # convert spatial units to mm using info in attrs
    dataset = convert_spatial_units(
        dataset,
        to_units="mm",
        names=["body_positions", "pose_positions", "pose_positions_allo"],
    )

    # save command line args
    dataset.attrs = {
        "video_filename": str(Path(root, dat_path, datename, f"{datename}.mp4")),
        "datename": datename,
        "root": root,
        "dat_path": dat_path,
        "res_path": res_path,
        "sampling_rate_Hz": sampling_rate,
        "target_sampling_rate_Hz": target_sampling_rate,
        "ref_time": ref_time,
    }

    filepath_swap = Path(root, res_path, datename, f"{datename}_idswaps.txt")
    if fix_fly_indices and os.path.exists(filepath_swap):
        logger.info(f"   Applying fly identity fixes from {filepath_swap}.")
        try:
            indices, flies1, flies2 = ld.load_swap_indices(filepath_swap)
            dataset = ld.swap_flies(dataset, indices, flies1=flies1, flies2=flies2)
            dataset.attrs["swap_events"] = [[ii, f1, f2] for ii, f1, f2 in zip(indices, flies1, flies2)]
            logger.info(f"  Fixed fly identities using info from {filepath_swap}.")
        except (FileNotFoundError, OSError) as e:
            logger.debug(f"      Could not load fly identities using info from {filepath_swap}.")
            logger.debug(e)
    logger.info("Done.")

    return dataset


def add_time(ds, ss, dim: str = "frame_number", suffix: str = ""):
    ds = ds.assign_coords({"frametimes" + suffix: (dim, ss.frame_time(frame=ds[dim]))})
    ds = ds.assign_coords({"framesamples" + suffix: (dim, ss.sample(frame=ds[dim]))})
    return ds


def _interp(ds, dim, target_frames_float, interp_kwargs={}):
    new_shape = list(ds.shape)
    new_shape[0] = len(target_frames_float)

    # interp for first item to get interp all coords correctly
    ds_new = ds[:, :1, ...].interp({dim: target_frames_float}, assume_sorted=True, kwargs=interp_kwargs)

    # but now the first dim is size 1 - so we need to fix this
    coords = ds_new.coords
    # maps for the too short dims: dim name - ref dim
    coords_map = {
        "flies": "flies",
        "chambers": "flies",
        "data_ball": "data_ball",
        "params_movie": "params_movie",
    }
    new_coords = {}
    for key in coords_map.keys():
        if key in coords:
            new_coords[key] = ds.coords[key].data  # keep the data from the too short dim
            del coords[key]  # remove the too short dim

    # make new DataArray missing values for the too short dims
    ds_x = xr.DataArray(data=np.empty(new_shape), coords=coords)
    ds_x.attrs = ds.attrs.copy()

    # add back in the too short dims from the original ds
    for key in new_coords.keys():
        ds_x.coords[key] = ((coords_map[key]), new_coords[key])

    # no need to run interp for the first dim again
    ds_x.data[:, 0, ...] = ds_new.data[:, 0, ...]
    del ds_new

    # now interp all remaining dims
    for fly in range(1, new_shape[1]):
        interpolator = scipy.interpolate.interp1d(
            ds[dim].data,
            ds.data[:, fly, ...],
            axis=0,
            bounds_error=False,
            assume_sorted=True,
            **interp_kwargs,
        )
        ds_x.data[:, fly, ...] = interpolator(target_frames_float)

    return ds_x


def align_time(
    ds,
    ss,
    target_samples,
    target_time,
    ref_time,
    dim: str = "frame_number",
    suffix: str = "",
    time=None,
    extrapolate: bool = False,
):
    target_frames_float = ss.times2frames(ss.sample_time(target_samples))
    interp_kwargs = {}
    if extrapolate:
        interp_kwargs["fill_value"] = "extrapolate"

    ds = _interp(ds, dim, target_frames_float, interp_kwargs)
    # ds = ds.interp({dim: target_frames_float}, assume_sorted=True, kwargs=interp_kwargs)

    ds = ds.drop_vars(
        ["frame_number" + suffix, "frame_times" + suffix, "frame_samples" + suffix],
        errors="Ignore",
    )

    # time_new = ds['frametimes' + suffix] - ref_time
    ds = ds.assign_coords({"time": ((dim), target_time)})
    ds = ds.swap_dims({dim: "time"})
    ds = ds.assign_coords(
        {
            "nearest_frame"
            + suffix: (
                ("time"),
                np.round(target_frames_float).astype(np.intp),
            )
        }
    )

    fps = 1 / np.nanmean(np.diff(ds["frametimes" + suffix]))
    ds.attrs.update({"video_fps": fps})

    ds = ds.drop_vars(
        [
            "frame_number" + suffix,
            "frame_times" + suffix,
            "frame_samples" + suffix,
            "framenumber" + suffix,
            "frametimes" + suffix,
            "framesamples" + suffix,
        ],
        errors="Ignore",
    )

    return ds


def assemble_metrics(
    dataset,
    make_abs: bool = True,
    make_rel: bool = True,
    smooth_positions: bool = True,
    use_true_times: bool = False,
    custom_pose_names: Optional[Dict[str, str]] = None,
):
    """[summary]

    Args:
        dataset ([type]): [description]
        make_abs (bool, optional): [description]. Defaults to True.
        make_rel (bool, optional): [description]. Defaults to True.
        smooth_positions (bool, optional): [description]. Defaults to True.
        use_true_times (bool, optional): Will use times for each frame from the data as dt
                                         for speed and acceleration calculations.
                                         Defaults to False (timestep=1).
        custom_pose_names (Dict[str, str]): Dictionary mapping default pose names (head, thorax, left_wing, right_wing) to custom names.

    Returns:
        [type]: [description]
        xarray.Dataset: containing these features:
                angles [time,flies]
                vels [time,flies,forward/lateral]
                chamber_vels [time,flies,y/x]
                rotational_speed [time,flies]
                accelerations [time,flies,forward/lateral]
                chamber_acc [time,flies,y/x]
                rotational_acc [time,flies]
                wing_angle_left [time,flies]
                wing_angle_right [time,flies]
                wing_angle_sum [time,flies]
                relative angle [time,flies,flies]
                (relative orientation) [time,flies,flies]
                (relative velocities) [time,flies,flies,y/x]
    """

    time = dataset.time.data
    nearest_frame = dataset.nearest_frame.data

    pose_names = {
        "thorax": "thorax",
        "head": "head",
        "left_wing": "left_wing",
        "right_wing": "right_wing",
    }
    if custom_pose_names is not None:
        pose_names.update(custom_pose_names)

    if use_true_times:
        timestep = dataset.time.data
    else:
        timestep = 1

    if "pose_positions_allo" in dataset:
        thoraces = dataset.pose_positions_allo.loc[:, :, pose_names["thorax"], :].values.astype(np.float32)
        heads = dataset.pose_positions_allo.loc[:, :, pose_names["head"], :].values.astype(np.float32)
        wing_left = dataset.pose_positions_allo.loc[:, :, pose_names["left_wing"], :].values.astype(np.float32)
        wing_right = dataset.pose_positions_allo.loc[:, :, pose_names["right_wing"], :].values.astype(np.float32)
        sampling_rate = dataset.pose_positions.attrs["sampling_rate_Hz"]
        frame_rate = dataset.pose_positions.attrs["video_fps"]
    elif "body_positions" in dataset:
        logger.warning(
            "No pose tracking data in dataset. Trying standard tracking data to compute metrics. Metrics may be wrong/incomplete."
        )
        thoraces = dataset.body_positions.loc[:, :, "center", :].values.astype(np.float32)
        heads = dataset.body_positions.loc[:, :, "head", :].values.astype(np.float32)
        wing_left = np.zeros_like(thoraces)
        wing_right = np.zeros_like(thoraces)
        wing_left[:] = np.nan
        wing_right[:] = np.nan
        sampling_rate = dataset.body_positions.attrs["sampling_rate_Hz"]
        frame_rate = dataset.body_positions.attrs["video_fps"]
    else:
        raise ValueError("No tracking data in dataset.")

    if smooth_positions:
        # Smoothing window should span 2 frames to get smooth acceleration traces.
        # Since std of Gaussian used for smoothing has std of winlen/8, winlen should span 16 frames.
        winlen = np.ceil(16 / frame_rate * sampling_rate)
        thoraces = mt.smooth(thoraces, winlen)
        heads = mt.smooth(heads, winlen)
        wing_left = mt.smooth(wing_left, winlen)
        wing_right = mt.smooth(wing_right, winlen)

    angles = mt.angle(thoraces, heads)
    chamber_vels = mt.velocity(thoraces, ref="chamber")

    ds_dict = dict()
    if make_abs:
        vels = mt.velocity(thoraces, heads, timestep=timestep)
        accelerations = mt.acceleration(thoraces, heads, timestep=timestep)
        chamber_acc = mt.acceleration(thoraces, ref="chamber", timestep=timestep)

        vels_x = chamber_vels[..., 1]
        vels_y = chamber_vels[..., 0]
        vels_forward = vels[..., 0]
        vels_lateral = vels[..., 1]
        vels_mag = np.linalg.norm(vels, axis=2)
        accs_x = chamber_acc[..., 1]
        accs_y = chamber_acc[..., 0]
        accs_forward = accelerations[..., 0]
        accs_lateral = accelerations[..., 1]
        accs_mag = np.linalg.norm(accelerations, axis=2)

        rotational_speed = mt.rot_speed(thoraces, heads, timestep=timestep)
        rotational_acc = mt.rot_acceleration(thoraces, heads, timestep=timestep)

        wing_angle_left = 180 - mt.internal_angle(wing_left, thoraces, heads)
        wing_angle_right = 180 - mt.internal_angle(wing_right, thoraces, heads)
        wing_angle_sum = wing_angle_left + wing_angle_right

        list_absolute = [
            angles,
            rotational_speed,
            rotational_acc,
            vels_mag,
            vels_x,
            vels_y,
            vels_forward,
            vels_lateral,
            accs_mag,
            accs_x,
            accs_y,
            accs_forward,
            accs_lateral,
            wing_angle_left,
            wing_angle_right,
            wing_angle_sum,
        ]

        abs_feature_names = [
            "angles",
            "rotational_speed",
            "rotational_acceleration",
            "velocity_magnitude",
            "velocity_x",
            "velocity_y",
            "velocity_forward",
            "velocity_lateral",
            "acceleration_mag",
            "acceleration_x",
            "acceleration_y",
            "acceleration_forward",
            "acceleration_lateral",
            "wing_angle_left",
            "wing_angle_right",
            "wing_angle_sum",
        ]

        absolute = np.stack(list_absolute, axis=2)

        ds_dict["abs_features"] = xr.DataArray(
            data=absolute.data,
            dims=["time", "flies", "absolute_features"],
            coords={
                "time": time,
                "absolute_features": abs_feature_names,
                "nearest_frame": (("time"), nearest_frame),
            },
            attrs={
                "description": 'coords are "egocentric" - rel. to box',
                "sampling_rate_Hz": sampling_rate,
                "time_units": "seconds",
                "spatial_units": "pixels",
            },
        )
    if make_rel:
        # RELATIVE FEATURES #
        dis = mt.distance(thoraces)
        rel_angles = mt.relative_angle(thoraces, heads)
        rel_orientation = angles[:, np.newaxis, :] - angles[:, :, np.newaxis]
        rel_velocities_y = chamber_vels[..., 0][:, np.newaxis, :] - chamber_vels[..., 0][:, :, np.newaxis]
        rel_velocities_x = chamber_vels[..., 1][:, np.newaxis, :] - chamber_vels[..., 1][:, :, np.newaxis]
        rel_velocities_lateral, rel_velocities_forward = mt.project_velocity(
            rel_velocities_x, rel_velocities_y, np.radians(angles)
        )
        rel_velocities_mag = np.sqrt(rel_velocities_forward**2 + rel_velocities_lateral**2)

        list_relative = [
            dis,
            rel_angles,
            rel_orientation,
            rel_velocities_mag,
            rel_velocities_forward,
            rel_velocities_lateral,
        ]

        rel_feature_names = [
            "distance",
            "relative_angle",
            "relative_orientation",
            "relative_velocity_mag",
            "relative_velocity_forward",
            "relative_velocity_lateral",
        ]

        relative = np.stack(list_relative, axis=3)
        ds_dict["rel_features"] = xr.DataArray(
            data=relative.data,
            dims=["time", "flies", "relative_flies", "relative_features"],
            coords={
                "time": time,
                "relative_features": rel_feature_names,
                "nearest_frame": (("time"), nearest_frame),
            },
            attrs={
                "description": 'coords are "egocentric" - rel. to box',
                "sampling_rate_Hz": sampling_rate,
                "time_units": "seconds",
                "spatial_units": "pixels",
            },
        )

    # MAKE ONE DATASET
    feature_dataset = xr.Dataset(ds_dict, attrs={})
    return feature_dataset


def convert_spatial_units(ds: xr.Dataset, to_units: Optional[str] = None, names: Optional[List[str]] = None) -> xr.Dataset:
    """px <-> mm using attrs['pixels_to_mm'] and attrs['spatial_units'].

    Args:
        ds (xr.Dataset): xb Dataset to process
        to_units (Optional[str], optional): Target units - either 'pixels' or 'mm'. Defaults to None (always convert mm <-> px).
        names (Optional[List[str]], optional): Names of DataArrays/sets in ds to convert. Defaults to None.

    Returns:
        xr.Dataset: with converted spatial units
    """
    if names is None:
        names = ["body_positions", "pose_positions", "pose_positions_allo"]

    for name in names:
        if name in ds:
            da = ds[name]
            spatial_units = da.attrs["spatial_units"]
            pixel_size_mm = da.attrs["pixel_size_mm"]

            if pixel_size_mm is np.nan:  # can't convert so skip
                continue

            if to_units is not None and spatial_units == to_units:  # already in the correct units so skip
                continue

            if spatial_units == "mm":  # convert to pixels
                da /= pixel_size_mm
                da.attrs["spatial_units"] = "pixels"
            elif spatial_units == "pixels":  # convert to mm
                da *= pixel_size_mm
                da.attrs["spatial_units"] = "mm"
    return ds


def interp_duplicates(y: np.array) -> np.array:
    """Replace duplicates with linearly interpolated values.

    Args:
        y (np.array): array with duplicates

    Returns:
        np.array: array without duplicates
    """
    x = np.arange(len(y))

    uni_mask = np.zeros_like(y, dtype=bool)
    uni_mask[np.unique(y, return_index=True)[1]] = True

    interp = scipy.interpolate.interp1d(x[uni_mask], y[uni_mask], kind="linear", copy=False, fill_value="extrapolate")

    y[~uni_mask] = interp(x[~uni_mask])

    return y


def save(savepath, dataset):
    """[summary]

    Args:
        savepath ([type]): [description]
        dataset ([type]): [description]
    """
    # xarray can't save attrs with dict values.
    # Delete event_times prior to saving.
    # Should make event_times a sparse xr.DataArray in the future.
    if "song_events" in dataset:
        if "event_times" in dataset.song_events.attrs:
            del dataset.song_events.attrs["event_times"]

    with zarr.ZipStore(savepath, mode="w") as zarr_store:
        # re-chunking does not seem to help with IO speed upon lazy loading
        chunks = dict(dataset.dims)
        chunks["time"] = 100_000
        chunks["sampletime"] = 100_000
        dataset = dataset.chunk(chunks)
        dataset.to_zarr(store=zarr_store, compute=True)


def _normalize_strings(dataset):
    """Ensure all keys in coords are proper (unicode?) python strings, not byte strings."""
    for key, val in dataset.coords.items():
        if val.dtype == "S16":
            dataset[key] = [v.decode() for v in val.data]
    return dataset


def load(savepath, lazy: bool = False, normalize_strings: bool = True, use_temp: bool = False):
    """[summary]

    Args:
        savepath ([type]): [description]
        lazy (bool, optional): [description]. Defaults to True.
        normalize_strings (bool, optional): [description]. Defaults to True.
        use_temp (bool, optional): Unpack zip to temp file - potentially speeds up loading and allows overwriting existing zarr file.
                                   Defaults to True.
    Returns:
        [type]: [description]
    """
    zarr_store = zarr.ZipStore(savepath, mode="r")
    if use_temp:
        dest = zarr.TempStore()
        zarr.copy_store(zarr_store, dest)
        zarr_store.close()
        zarr_store = dest
    dataset = xr.open_zarr(zarr_store)
    if not lazy:
        dataset.load()
        zarr_store.close()

    if normalize_strings:
        dataset = _normalize_strings(dataset)
    logger.info(dataset)
    return dataset
