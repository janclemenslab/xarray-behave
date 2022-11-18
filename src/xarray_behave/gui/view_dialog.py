import sys
import numpy as np
from qtpy import QtWidgets, QtCore
from superqt import QLabeledDoubleSlider, QLabeledDoubleRangeSlider

# style sheet required fix broken QSlider in Macos (https://bugreports.qt.io/browse/QTBUG-98093)
QSS = """

QSlider::groove:horizontal {
    background: #c4c4c4;
    height: 12px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #5555ff;
    border: 0px solid #5c5c5c;
    width: 8px;
    border-radius: 2px;
}
"""


class TextSlider(QtWidgets.QWidget):

    def __init__(self,
                 min_value: float,
                 max_value: float,
                 description: str,
                 model,
                 attr_name: str,
                 default_value: float = 0.0,
                 checkable: bool = True):
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
            self.sld = QLabeledDoubleRangeSlider(parent=self)
        else:
            self.sld = QLabeledDoubleSlider(parent=self)
        # style sheet required fix broken QSlider in Macos (https://bugreports.qt.io/browse/QTBUG-98093)
        self.sld.setStyleSheet(QSS)

        self.sld.setOrientation(QtCore.Qt.Horizontal)
        self.sld.setRange(self.min_value, self.max_value)
        self.sld.setDecimals(self.decimals)
        self.sld.setSingleStep(10.0**(-self.decimals))
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
            [
                self.model.__setattr__(a, v)
                for a, v in zip(self.attr_name, self.value)
            ]
        else:
            self.model.__setattr__(self.attr_name, self.value)


class QHSeperationLine(QtWidgets.QFrame):

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(1)
        self.setFixedHeight(20)
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                           QtWidgets.QSizePolicy.Minimum)


class Form(QtWidgets.QDialog):

    def __init__(self, parent=None, model=None):
        super(Form, self).__init__(parent)
        self.setWindowTitle("Spectrogram view params")
        self.model = model
        self.setFixedWidth(800)

        layout = QtWidgets.QVBoxLayout(self)

        y_max = np.max(self.model.y)

        if self.model.vr:
            layout.addWidget(QtWidgets.QLabel('<b>Video</b>'))

            layout.addWidget(
                TextSlider(min_value=2,
                           max_value=max(self.model.vr.frame_shape) // 2,
                           default_value=None,
                           description='Video crop',
                           model=self.model,
                           checkable=False,
                           attr_name='box_size'))
            layout.addWidget(QHSeperationLine())

        layout.addWidget(QtWidgets.QLabel('<b>Waveform</b>'))
        layout.addWidget(
            TextSlider(min_value=0,
                       max_value=y_max * 2,
                       default_value=y_max,
                       description='Waveform vertical limits',
                       model=self.model,
                       attr_name='ylim'))
        layout.addWidget(QHSeperationLine())

        layout.addWidget(QtWidgets.QLabel('<b>Spectrogram</b>'))
        layout.addWidget(
            TextSlider(min_value=0.0,
                       max_value=self.model.fs_song / 2,
                       default_value=[0, self.model.fs_song / 2],
                       description='Spectrogram frequency limits [Hz]',
                       model=self.model,
                       attr_name=['fmin', 'fmax']))

        s_max = np.nanmax(self.model.spec_view.S)
        layout.addWidget(
            TextSlider(min_value=0,
                       max_value=s_max * 2,
                       default_value=[None, None],
                       description='Spectrogram color limits',
                       model=self.model,
                       attr_name='spec_levels'))

        layout.addWidget(
            TextSlider(min_value=0,
                       max_value=64,
                       description='Spectrogram color compression',
                       default_value=0.0,
                       checkable=False,
                       model=self.model,
                       attr_name='spec_compression_ratio'))

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
