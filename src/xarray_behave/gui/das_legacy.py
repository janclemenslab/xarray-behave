"""DAS 0.33 dialogs, retained for releases without das.gui_app."""

import logging
import multiprocessing
import os
import pathlib
import threading
from typing import Any, Dict

import numpy as np
import rich
import yaml
from qtpy import QtCore, QtWidgets

import xarray_behave
from . import utils
from .formbuilder import YamlDialog

logger = logging.getLogger(__name__)
package_dir = xarray_behave.__path__[0]

def das_train(self, qt_keycode):
    def _filter_form_data(form_data: Dict[str, Any], is_cli: bool = False) -> Dict[str, Any]:
        """[summary]

        Args:
            form_data (Dict[Any]): [description]
            cli (bool, optional): Process boolean flags. Defaults to False.

        Returns:
            Dict[Any]: [description]
        """

        form_data["model_name"] = "tcn_stft"
        if form_data["frontend"] == "STFT":
            form_data["nb_pre_conv"] = int(np.sqrt(int(form_data["nb_pre_conv"])))
            form_data["pre_nb_dft"] = int(form_data["pre_nb_dft"])
        else:
            form_data["nb_pre_conv"] = int(0)
        del form_data["frontend"]
        form_data["reduce_lr"] = form_data["reduce_lr_patience"] is not None
        post_opt = form_data["postopt"] == "Yes"
        form_data = {k: v for k, v in form_data.items() if v not in ["Yes", "No", None]}
        form_data["post_opt"] = post_opt

        if not len(form_data["y_suffix"]):
            del form_data["y_suffix"]
        if not len(form_data["save_prefix"]):
            del form_data["save_prefix"]

        for field in ["seed", "fraction_data", "reduce_lr_patience"]:
            if field in form_data and form_data[field] is None:
                del form_data[field]
        form_data["use_separable"] = [item.lower() == "true" for item in form_data["use_separable"]]
        if is_cli:
            form_data["use_separable"] = " ".join([str(item) for item in form_data["use_separable"]])
            for field in ["balance", "ignore_boundaries", "reduce_lr", "tensorboard", "post_opt"]:
                if field in form_data:
                    if form_data[field] is False:  # rename and pre-prend "no_"
                        del form_data[field]
                        form_data["no_" + field] = ""
                    else:  # empyt value
                        form_data[field] = ""

        return form_data

    def save(arg):
        savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save configuration to", "", filter="yaml files (*.yaml);;all files (*)"
        )
        if len(savefilename):
            data = dialog.form.get_form_data()
            logger.info(f"   Saving form fields to {savefilename}.")
            with open(savefilename, "w") as stream:
                yaml.safe_dump(data, stream)
            logger.info("Done.")

    def make_cli(arg):
        script_ext = "cmd" if os.name == "nt" else "sh"
        savefilename = "train." + script_ext
        savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Script name", savefilename, filter=f"script (*.{script_ext};;all files (*)"
        )
        if len(savefilename):
            form_data = dialog.form.get_form_data()
            form_data = _filter_form_data(form_data, is_cli=True)

            cmd = "python3 -m das.train"
            # FIXME formatting
            for key, val in form_data.items():
                cmd += f" --{key.replace('_', '-')} {val}"
            with open(savefilename, "w") as f:
                f.write(cmd)

            # TODO: dialog to suggest editing the script (change paths, activate conda env, cluster specific stuff)
            logger.info("Done.")

    def load(arg):
        yaml_filename, _ = QtWidgets.QFileDialog.getOpenFileName(parent=None, caption="Select yaml file")
        if len(yaml_filename):
            logger.info(f"   Updating form fields with information from {yaml_filename}.")
            with open(yaml_filename, "r") as stream:
                data = yaml.safe_load(stream)
            dialog.form.set_form_data(data)  # update form
            logger.info("Done.")

    data_dir = QtWidgets.QFileDialog.getExistingDirectory(None, caption="Open Data Folder (*.npy)")
    # TODO: check that this is a valid data_dir!
    dialog = YamlDialog(
        yaml_file=package_dir + "/gui/forms/das_train.yaml",
        title="Train network",
        # main_callback=train,
        callbacks={"save": save, "load": load, "make_cli": make_cli},
    )
    dialog.form.fields["data_dir"].setText(data_dir)
    dialog.form.fields["save_dir"].setText(os.path.splitext(data_dir)[0] + ".res")
    y_suffices = list(set([p.stem.strip("y_") for p in pathlib.Path(data_dir).glob("**/y_*.npy")]))
    dialog.form.fields["y_suffix"].set_options(y_suffices)

    dialog.show()
    result = dialog.exec_()
    if result == QtWidgets.QDialog.Accepted:
        form_data = dialog.form.get_form_data()
        form_data = _filter_form_data(form_data)

        got_das = False
        try:
            import das.train

            got_das = True
        except ImportError:
            logger.exception("Need to install das. Alternatively, make scripts and run training elsewhere.")

        if got_das:
            # start in independent process, otherwise GUI will freeze during training
            form_data["log_messages"] = True

            queue = multiprocessing.Queue()  # for progress updates from the callback for display in the GUI
            stop_event = threading.Event()  # checked in the keras callback for stopping training from the GUI

            progress = QtWidgets.QProgressDialog("Initializing training", "Stop training", 0, form_data["nb_epoch"], None)
            progress.setWindowTitle("DAS training")
            progress.setWindowModality(QtCore.Qt.NonModal)

            # Custom cancel-button callback the sets the stop_event to stop training."""
            def custom_cancel():
                utils.invoke_in_main_thread(progress.setLabelText, "Stopping training.")
                stop_event.set()  # stop training

            progress.canceled.connect(custom_cancel)

            form_data["_qt_progress"] = (queue, stop_event)
            worker_training = utils.Worker(das.train.train, **form_data)

            pool = QtCore.QThreadPool.globalInstance()

            def update_progress(queue):
                while True:
                    value = queue.get()

                    if value[0] is None or value[0] < 0 or pool.activeThreadCount() == 1:
                        utils.invoke_in_main_thread(progress.cancel)
                        utils.invoke_in_main_thread(progress.close)
                        break  # stop this thread
                    else:
                        utils.invoke_in_main_thread(progress.setValue, value[0])
                        utils.invoke_in_main_thread(progress.setLabelText, str(value[1]))
                    QtWidgets.QApplication.processEvents()

            worker_progress = utils.Worker(update_progress, queue)
            pool.start(worker_progress)
            pool.start(worker_training)
            QtWidgets.QApplication.processEvents()

def das_predict(self, qt_keycode):
    logger.info("Predicting song using DAS:")

    if hasattr(self, "ds") and "song_raw" not in self.ds:
        logger.error("   Missing `song_raw`. skipping.")

    try:
        import das.predict
        import das.utils
    except ImportError as e:
        logger.exception(e)
        logger.info("   Failed to import das. Install via `pip install das`.")
        return

    dialog = YamlDialog(yaml_file=package_dir + "/gui/forms/das_predict.yaml", title="Predict labels using DAS")

    # get model path and populate form
    model_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, caption="Open model file (*_model.h5)")
    if len(model_path):
        dialog.form["model_path"] = model_path
    else:
        logger.warning("No model selected.")
        return

    # populate values for postprocessing from params if they exist
    model_path = model_path.rsplit("_", 1)[0]  # split off suffix (_model.h5 or params.yaml)
    params = das.utils.load_params(model_path)
    if "post_opt" in params and isinstance(params["post_opt"], dict):
        dialog.form["segment_fillgap"] = params["post_opt"]["gap_dur"]
        dialog.form["segment_minlen"] = params["post_opt"]["min_len"]

    # if no file loaded, ask to select file or folder
    if not hasattr(self, "ds"):
        dialog.form["file"] = "Select file or folder"

    dialog.show()
    result = dialog.exec_()

    if result == QtWidgets.QDialog.Accepted:
        form_data = dialog.form.get_form_data()
        model_path = form_data["model_path"]
        if model_path == " ":
            logger.warning("No model for prediction selected.")
            return
        else:
            model_path = model_path.rsplit("_", 1)[0]  # split off suffix

        # set these to 0 to avoid post processing - none will use values from params file
        if form_data["segment_fillgap"] is None:
            form_data["segment_fillgap"] = 0
        if form_data["segment_minlen"] is None:
            form_data["segment_minlen"] = 0

        if form_data["filter_song"] == "no":
            form_data["bandpass_low_freq"] = None
            form_data["bandpass_up_freq"] = None

        if form_data["file"] != "Current file":
            if form_data["folder"] != " ":
                file_path = form_data["folder"]
            elif form_data["file"] != " ":
                file_path = form_data["file"]
            else:
                logger.warning("No audio data. Either open a file or select a folder or file in the predict dialog.")
                return

            das.predict.cli_predict(
                file_path,
                model_path,
                verbose=1,
                batch_size=32,
                save_format=form_data["save_format"],
                event_thres=form_data["event_thres"],
                event_dist=form_data["event_dist"],
                event_dist_min=form_data["event_dist_min"],
                event_dist_max=form_data["event_dist_max"],
                segment_thres=form_data["event_thres"],
                segment_fillgap=form_data["segment_fillgap"],
                segment_minlen=form_data["segment_minlen"],
                bandpass_low_freq=form_data["bandpass_low_freq"],
                bandpass_up_freq=form_data["bandpass_up_freq"],
                resample=form_data["resample"],
            )
            return
        elif hasattr(self, "ds") and form_data["file"] == "Current file":
            start_seconds = form_data["start_seconds"]
            start_index = utils.find_nearest_idx(self.ds.sampletime.values, start_seconds)

            end_seconds = form_data["end_seconds"]
            end_index = end_seconds
            if end_seconds is not None:
                end_index = utils.find_nearest_idx(self.ds.sampletime.values, end_seconds)

            audio = self.ds.song_raw.data[start_index:end_index]

            params = das.utils.load_params(model_path)
            if audio.shape[0] < params["nb_hist"]:
                logger.warning(
                    f"   Aborting. Audio has fewer samples ({audio.shape[0]}) shorter    than network chunk size ({params['nb_hist']}).    Fix by select longer audio."
                )
                return

            # select batch size so that at least 10 batches are run
            # minimizes loss of annotations from batch size "quantization" errors
            batch_size = 32
            nb_batches = int(
                np.floor((audio.shape[0] - ((batch_size - 1) + params["nb_hist"])) / (params["stride"] * (batch_size)))
            )
            while nb_batches < 10 and batch_size > 1:
                batch_size -= 1

            logger.info("   Running inference on audio.")
            logger.info(f"   Model from {model_path}.")
            params = das.utils.load_params(model_path)

            rich.print(form_data)
            events, segments, _, _ = das.predict.predict(
                audio,
                model_path,
                verbose=1,
                batch_size=batch_size,
                event_thres=form_data["event_thres"],
                event_dist=form_data["event_dist"],
                event_dist_min=form_data["event_dist_min"],
                event_dist_max=form_data["event_dist_max"],
                segment_thres=form_data["event_thres"],
                segment_fillgap=form_data["segment_fillgap"],
                segment_minlen=form_data["segment_minlen"],
                bandpass_low_freq=form_data["bandpass_low_freq"],
                bandpass_up_freq=form_data["bandpass_up_freq"],
                resample=form_data["resample"],
                fs_audio=self.fs_song,
            )

            # Process detected song
            if not events and not segments:
                logger.warning("Found no song.")
                return

            suffix = ""
            if form_data["proof_reading_mode"]:
                suffix = "_proposals"

            # events['seconds'] is in samples/fs - translate to time stamps via sample time
            detected_event_names = []
            if "sequence" in events:
                detected_event_names = np.unique(events["sequence"])

            if len(detected_event_names) > 0 and detected_event_names[0] is not None:
                logger.info(f"   found {len(events['seconds'])} instances of events '{detected_event_names}'.")
                event_samples = (np.array(events["seconds"]) * self.fs_song + start_index).astype(np.uintp)

                # make sure all detected events are within bounds
                event_samples = event_samples[event_samples >= 0]
                event_samples = event_samples[event_samples < self.ds.sampletime.shape[0]]

                event_seconds = self.ds.sampletime[event_samples]
                for name_or_index, seconds in zip(events["sequence"], event_seconds):
                    if type(name_or_index) is int:
                        event_name = str(events["names"][name_or_index])
                    else:
                        event_name = str(name_or_index)
                    self.event_times.add_time(event_name + str(suffix), seconds, seconds, category="event")

            detected_segment_names = []
            if "sequence" in segments:  # and len(segments['sequence']):  # and segments['sequence'] is not None:
                # segments['sequence'] = [s for s in segments['sequence'] if s is not None]
                detected_segment_names = np.unique(segments["sequence"])
                # if these are indices, get corresponding names
                if (
                    len(detected_segment_names)
                    and type(detected_segment_names[0]) is not str
                    and type(detected_segment_names[0]) is not np.str_
                ):
                    detected_segment_names = [segments["names"][ii] for ii in detected_segment_names]

            if len(detected_segment_names) > 0:  # and detected_segment_names[0] is not None:
                logger.info(f"   found {len(segments['onsets_seconds'])} instances of segments '{detected_segment_names}'.")
                onsets_samples = (np.array(segments["onsets_seconds"]) * self.fs_song + start_index).astype(np.uintp)
                offsets_samples = (np.array(segments["offsets_seconds"]) * self.fs_song + start_index).astype(np.uintp)

                # make sure all detected segments are within bounds
                onsets_samples = onsets_samples[onsets_samples >= 0]
                offsets_samples = offsets_samples[offsets_samples >= 0]
                onsets_samples = onsets_samples[onsets_samples < self.ds.sampletime.shape[0]]
                offsets_samples = offsets_samples[offsets_samples < self.ds.sampletime.shape[0]]

                onsets_seconds = self.ds.sampletime[onsets_samples]
                offsets_seconds = self.ds.sampletime[offsets_samples]
                for name_or_index, onset_seconds, offset_seconds in zip(
                    segments["sequence"], onsets_seconds, offsets_seconds
                ):
                    if type(name_or_index) is not str and type(detected_segment_names[0]) is not np.str_:
                        segment_name = segments["names"][name_or_index]
                    else:
                        segment_name = str(name_or_index)

                    self.event_times.add_time(segment_name + str(suffix), onset_seconds, offset_seconds, category="segment")

            self.nb_eventtypes = len(self.event_times)
            self.eventtype_colors = utils.make_colors(self.nb_eventtypes)
            self.update_eventtype_selector()
    logger.info("Done.")
