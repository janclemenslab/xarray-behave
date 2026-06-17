import sys
import numpy as np
from qtpy import QtWidgets, QtCore

SPECTROGRAM_COLORMAPS = (
    "turbo",
    "viridis",
    "magma",
    "inferno",
    "plasma",
    "cividis",
    "gray",
    "bone",
    "hot",
    "cool",
)


def _spectrogram_level_max(model):
    spec_view = getattr(model, "spec_view", None)
    S = getattr(spec_view, "S", None)
    if S is None:
        return 1.0
    values = np.asarray(S)
    finite_values = values[np.isfinite(values)]
    if not finite_values.size:
        return 1.0
    s_max = float(np.max(finite_values))
    if s_max <= 0:
        return 1.0
    return s_max * 2


def _make_colormap_row(model):
    row = QtWidgets.QWidget()
    label = QtWidgets.QLabel("Colormap", row)
    label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    label.setFixedWidth(240)

    combo = QtWidgets.QComboBox(row)
    combo.setObjectName("spectrogramColormapCombo")
    combo.addItems(SPECTROGRAM_COLORMAPS)
    current_colormap = getattr(model, "spec_colormap", SPECTROGRAM_COLORMAPS[0])
    if current_colormap not in SPECTROGRAM_COLORMAPS:
        combo.addItem(current_colormap)
    combo.setCurrentText(current_colormap)
    combo.currentTextChanged.connect(lambda value: setattr(model, "spec_colormap", value))

    hbox = QtWidgets.QHBoxLayout(row)
    hbox.setContentsMargins(0, 0, 0, 0)
    hbox.addWidget(label)
    hbox.addWidget(combo)
    return row, combo


class DoubleSliderControl(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimum = 0.0
        self._maximum = 1.0
        self._steps = 10_000
        self._updating = False

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, self._steps)
        self.spin = QtWidgets.QDoubleSpinBox(self)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)

    def setOrientation(self, orientation):
        self.slider.setOrientation(orientation)

    def setRange(self, minimum: float, maximum: float):
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self.spin.setRange(self._minimum, self._maximum)

    def setDecimals(self, decimals: int):
        self.spin.setDecimals(int(decimals))

    def setSingleStep(self, step: float):
        self.spin.setSingleStep(float(step))

    def value(self):
        return float(self.spin.value())

    def setValue(self, value):
        self._set_value(float(value), emit=True)

    def _position_for_value(self, value: float) -> int:
        span = self._maximum - self._minimum
        if span <= 0:
            return 0
        return int(round((value - self._minimum) / span * self._steps))

    def _value_for_position(self, position: int) -> float:
        span = self._maximum - self._minimum
        if span <= 0:
            return self._minimum
        return self._minimum + position / self._steps * span

    def _set_value(self, value: float, *, emit: bool):
        value = min(max(float(value), self._minimum), self._maximum)
        self._updating = True
        self.spin.setValue(value)
        self.slider.setValue(self._position_for_value(value))
        self._updating = False
        if emit:
            self.valueChanged.emit(self.value())

    def _on_slider_changed(self, position: int):
        if self._updating:
            return
        self._set_value(self._value_for_position(position), emit=True)

    def _on_spin_changed(self, value: float):
        if self._updating:
            return
        self._set_value(value, emit=True)


class DoubleRangeSliderControl(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimum = 0.0
        self._maximum = 1.0
        self._steps = 10_000
        self._updating = False

        self.min_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.max_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.min_slider.setRange(0, self._steps)
        self.max_slider.setRange(0, self._steps)
        self.min_spin = QtWidgets.QDoubleSpinBox(self)
        self.max_spin = QtWidgets.QDoubleSpinBox(self)

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(2)
        min_label = QtWidgets.QLabel("Min", self)
        max_label = QtWidgets.QLabel("Max", self)
        layout.addWidget(min_label, 0, 0)
        layout.addWidget(self.min_slider, 0, 1)
        layout.addWidget(self.min_spin, 0, 2)
        layout.addWidget(max_label, 1, 0)
        layout.addWidget(self.max_slider, 1, 1)
        layout.addWidget(self.max_spin, 1, 2)

        self.min_slider.valueChanged.connect(lambda position: self._on_slider_changed(0, position))
        self.max_slider.valueChanged.connect(lambda position: self._on_slider_changed(1, position))
        self.min_spin.valueChanged.connect(lambda value: self._on_spin_changed(0, value))
        self.max_spin.valueChanged.connect(lambda value: self._on_spin_changed(1, value))

    def setOrientation(self, orientation):
        self.min_slider.setOrientation(orientation)
        self.max_slider.setOrientation(orientation)

    def setRange(self, minimum: float, maximum: float):
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self.min_spin.setRange(self._minimum, self._maximum)
        self.max_spin.setRange(self._minimum, self._maximum)

    def setDecimals(self, decimals: int):
        self.min_spin.setDecimals(int(decimals))
        self.max_spin.setDecimals(int(decimals))

    def setSingleStep(self, step: float):
        self.min_spin.setSingleStep(float(step))
        self.max_spin.setSingleStep(float(step))

    def value(self):
        return [float(self.min_spin.value()), float(self.max_spin.value())]

    def setValue(self, value):
        min_value, max_value = value
        self._set_values(float(min_value), float(max_value), emit=True)

    def _position_for_value(self, value: float) -> int:
        span = self._maximum - self._minimum
        if span <= 0:
            return 0
        return int(round((value - self._minimum) / span * self._steps))

    def _value_for_position(self, position: int) -> float:
        span = self._maximum - self._minimum
        if span <= 0:
            return self._minimum
        return self._minimum + position / self._steps * span

    def _set_values(self, min_value: float, max_value: float, *, emit: bool):
        min_value = min(max(float(min_value), self._minimum), self._maximum)
        max_value = min(max(float(max_value), self._minimum), self._maximum)
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        self._updating = True
        self.min_spin.setValue(min_value)
        self.max_spin.setValue(max_value)
        self.min_slider.setValue(self._position_for_value(min_value))
        self.max_slider.setValue(self._position_for_value(max_value))
        self._updating = False
        if emit:
            self.valueChanged.emit(self.value())

    def _on_slider_changed(self, index: int, position: int):
        if self._updating:
            return
        values = self.value()
        values[index] = self._value_for_position(position)
        self._set_values(values[0], values[1], emit=True)

    def _on_spin_changed(self, index: int, value: float):
        if self._updating:
            return
        values = self.value()
        values[index] = value
        self._set_values(values[0], values[1], emit=True)


class TextSlider(QtWidgets.QWidget):
    def __init__(
        self,
        min_value: float,
        max_value: float,
        description: str,
        model,
        attr_name: str,
        default_value: float = 0.0,
        checkable: bool = True,
    ):
        super().__init__()

        self.max_value = max_value
        self.min_value = min_value

        self.decimals = np.log10(max_value - min_value)
        if self.decimals > 0:
            self.decimals = 2
        else:
            self.decimals = np.ceil(-self.decimals) + 3

        self.default_value = default_value
        self.description = description
        self.checkable = checkable
        self.model = model
        self.attr_name = attr_name
        if isinstance(self.attr_name, list):
            value = [self.model.__getattribute__(a) for a in self.attr_name]
        else:
            value = self.model.__getattribute__(self.attr_name)

        if value is None:
            value = self.default_value
        self.value = value

        self.initUI()

    def initUI(self):
        self.label = QtWidgets.QLabel(self.description, self)
        self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.label.setFixedWidth(240)
        if isinstance(self.value, list):
            self.sld = DoubleRangeSliderControl(parent=self)
        else:
            self.sld = DoubleSliderControl(parent=self)

        self.sld.setOrientation(QtCore.Qt.Horizontal)
        self.sld.setRange(self.min_value, self.max_value)
        self.sld.setDecimals(self.decimals)
        self.sld.setSingleStep(10.0 ** (-self.decimals))
        if isinstance(self.value, list):
            self.sld.setValue([self.min_value, self.max_value])
        self.sld.valueChanged.connect(self.updateValue)

        if self.checkable:
            self.checkbox = QtWidgets.QCheckBox("Auto", self)
            self.checkbox.stateChanged.connect(self.updateCheckBox)

            if isinstance(self.value, list):
                if None in self.value:
                    self.sld.setDisabled(True)
                    self.checkbox.setChecked(True)
                else:
                    self.sld.setValue(self.value)
                    self.value = self.sld.value()
            else:
                if self.value is None:
                    self.sld.setDisabled(True)
                else:
                    self.sld.setValue(self.value)
                    self.value = self.sld.value()
        elif not isinstance(self.value, list) and self.value is not None:
            self.sld.setValue(self.value)
            self.value = self.sld.value()

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.label)
        hbox.addWidget(self.sld)
        if self.checkable:
            hbox.addWidget(self.checkbox)

        self.setLayout(hbox)

    def updateValue(self, value):
        self.value = value
        self.update_model()

    def updateCheckBox(self):
        if self.checkbox.isChecked():
            self.sld.setDisabled(True)
            self.value = self.default_value
        else:
            self.sld.setDisabled(False)
            self.value = self.sld.value()
        self.update_model()

    def update_model(self):
        if isinstance(self.attr_name, list):
            [self.model.__setattr__(a, v) for a, v in zip(self.attr_name, self.value)]
        else:
            self.model.__setattr__(self.attr_name, self.value)


class SpectrogramResolutionControl(QtWidgets.QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self._last_value = 0

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(-4, 4)
        self.slider.setValue(0)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.valueChanged.connect(self._on_value_changed)

        left_label = QtWidgets.QLabel("Increase freq / decrease time", self)
        right_label = QtWidgets.QLabel("Decrease freq / increase time", self)
        left_label.setProperty("role", "muted")
        right_label.setProperty("role", "muted")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        labels = QtWidgets.QHBoxLayout()
        labels.addWidget(left_label)
        labels.addStretch(1)
        labels.addWidget(right_label)
        layout.addLayout(labels)
        layout.addWidget(self.slider)

    def _on_value_changed(self, value: int):
        delta = int(value) - self._last_value
        self._last_value = int(value)
        if delta < 0:
            for _ in range(abs(delta)):
                self.model.inc_freq_res(None)
        elif delta > 0:
            for _ in range(delta):
                self.model.dec_freq_res(None)


class QHSeperationLine(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(1)
        self.setFixedHeight(20)
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)


class SpectrogramSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, model=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrogram display settings")
        self.model = model
        self.setMinimumWidth(640)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("<b>Spectrogram</b>"))

        colormap_row, self.colormap_combo = _make_colormap_row(self.model)
        layout.addWidget(colormap_row)

        self.frequency_slider = TextSlider(
            min_value=0.0,
            max_value=self.model.fs_song / 2,
            default_value=[0, self.model.fs_song / 2],
            description="Frequency limits [Hz]",
            model=self.model,
            attr_name=["fmin", "fmax"],
        )
        layout.addWidget(self.frequency_slider)

        self.level_slider = TextSlider(
            min_value=0,
            max_value=_spectrogram_level_max(self.model),
            default_value=[None, None],
            description="Color limits",
            model=self.model,
            attr_name="spec_levels",
        )
        layout.addWidget(self.level_slider)

        self.compression_slider = TextSlider(
            min_value=0,
            max_value=64,
            description="Color compression",
            default_value=0.0,
            checkable=False,
            model=self.model,
            attr_name="spec_compression_ratio",
        )
        layout.addWidget(self.compression_slider)

        layout.addWidget(QtWidgets.QLabel("<b>Resolution</b>"))
        self.resolution_slider = SpectrogramResolutionControl(self.model, self)
        layout.addWidget(self.resolution_slider)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)


class Form(QtWidgets.QDialog):
    def __init__(self, parent=None, model=None):
        super(Form, self).__init__(parent)
        self.setWindowTitle("Spectrogram view params")
        self.model = model
        self.setFixedWidth(800)

        layout = QtWidgets.QVBoxLayout(self)

        y_max = np.max(self.model.y)

        if self.model.vr:
            layout.addWidget(QtWidgets.QLabel("<b>Video</b>"))

            layout.addWidget(
                TextSlider(
                    min_value=2,
                    max_value=max(self.model.vr.frame_shape) // 2,
                    default_value=None,
                    description="Video crop",
                    model=self.model,
                    checkable=False,
                    attr_name="box_size",
                )
            )
            layout.addWidget(QHSeperationLine())

        layout.addWidget(QtWidgets.QLabel("<b>Waveform</b>"))
        layout.addWidget(
            TextSlider(
                min_value=0,
                max_value=y_max * 2,
                default_value=y_max,
                description="Waveform vertical limits",
                model=self.model,
                attr_name="ylim",
            )
        )
        layout.addWidget(QHSeperationLine())

        layout.addWidget(QtWidgets.QLabel("<b>Spectrogram</b>"))
        colormap_row, _ = _make_colormap_row(self.model)
        layout.addWidget(colormap_row)
        layout.addWidget(
            TextSlider(
                min_value=0.0,
                max_value=self.model.fs_song / 2,
                default_value=[0, self.model.fs_song / 2],
                description="Spectrogram frequency limits [Hz]",
                model=self.model,
                attr_name=["fmin", "fmax"],
            )
        )

        layout.addWidget(
            TextSlider(
                min_value=0,
                max_value=_spectrogram_level_max(self.model),
                default_value=[None, None],
                description="Spectrogram color limits",
                model=self.model,
                attr_name="spec_levels",
            )
        )

        layout.addWidget(
            TextSlider(
                min_value=0,
                max_value=64,
                description="Spectrogram color compression",
                default_value=0.0,
                checkable=False,
                model=self.model,
                attr_name="spec_compression_ratio",
            )
        )

        chkbx_spec_denoise = QtWidgets.QCheckBox("Denoise spectrogram")

        def updateDenoiseCheckBox():
            self.model.spec_denoise = chkbx_spec_denoise.isChecked()

        chkbx_spec_denoise.stateChanged.connect(updateDenoiseCheckBox)
        layout.addWidget(chkbx_spec_denoise)

        chkbx_spec_mel = QtWidgets.QCheckBox("Mel spectrogram (not implemented)")

        def updateMelCheckBox():
            self.model.spec_mel = chkbx_spec_mel.isChecked()

        chkbx_spec_mel.stateChanged.connect(updateMelCheckBox)
        layout.addWidget(chkbx_spec_mel)

        self.setLayout(layout)
