import numpy as np
import pytest

import xarray_behave  # noqa: F401 - selects the Qt binding
from qtpy import QtCore, QtWidgets
from qtpy.QtTest import QTest
import pyqtgraph as pg

from xarray_behave.gui.event_widgets import WaveformPane


def test_opengl_waveform_annotations_and_threshold():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if app.platformName() == "offscreen":
        pytest.skip("OpenGL widgets require a desktop session")
    moves, thresholds, clicks = [], [], []
    widget = WaveformPane(
        region_changed_callback=lambda region: moves.append(region.getRegion()),
        callback=lambda *args: clicks.append(args),
    )
    widget.threshold_changed.connect(thresholds.append)
    widget.resize(900, 300)
    x = np.linspace(0, 1, 10_000)
    y = np.sin(50 * x)
    widget.set_waveform(x, y, y_other=np.column_stack([y * 0.7, y * 0.4]))
    widget.set_threshold_data(x, np.abs(y), enabled=True, threshold=0.4)
    widget.add_segment(0.2, 0.3, 0, brush=pg.mkBrush(0, 255, 0, 100), text="song")
    widget.add_event([0.7], 1, pg.mkPen("m", width=2), text="pulse")
    widget.show()
    widget.activateWindow()
    QTest.qWait(200)
    viewport = widget.viewport()
    try:
        assert viewport.context().isValid()
        assert widget._curve.curve.glstate is not None
        assert widget._threshold_curve.glstate is not None
        assert all(curve.glstate is not None for curve in widget._other_curves)

        def point(x, y):
            return widget.mapFromScene(widget.getViewBox().mapViewToScene(QtCore.QPointF(x, y)))

        def drag(start, stop):
            QTest.qWait(30)  # Let the scene's mouse rate limit expire before hovering.
            QTest.mouseMove(viewport, start)
            QTest.qWait(30)
            QTest.mousePress(viewport, QtCore.Qt.LeftButton, pos=start)
            for step in range(1, 6):
                QTest.mouseMove(viewport, start + (stop - start) * (step / 5), delay=20)
                app.processEvents()
            QTest.mouseRelease(viewport, QtCore.Qt.LeftButton, pos=stop)
            app.processEvents()

        drag(point(0.5, 0.4), point(0.5, 0.65))
        assert thresholds and abs(thresholds[-1] - 0.65) < 0.03
        drag(point(0.25, -0.6), point(0.4, -0.6))
        assert moves and np.allclose(moves[-1], [0.35, 0.45], atol=0.01)
        QTest.mouseClick(viewport, QtCore.Qt.LeftButton, pos=point(0.85, -0.5))
        assert clicks and abs(clicks[-1][0] - 0.85) < 0.01

        viewport.makeCurrent()
        width = round(viewport.width() * viewport.devicePixelRatioF())
        height = round(viewport.height() * viewport.devicePixelRatioF())
        pixels = np.empty((height, width, 4), dtype=np.uint8)
        viewport.context().extraFunctions().glReadPixels(0, 0, width, height, 0x1908, 0x1401, pixels.data)
        viewport.doneCurrent()
        assert np.count_nonzero((pixels[:, :, 1] > 80) & (pixels[:, :, 0] < 40)) > 100

        widget.clear_annotations()
        widget.set_threshold_data(None, None, enabled=False)
        # Exercise the independent peak-pair drawing path used when zoomed out.
        widget.set_waveform(x * 8, y)
        app.processEvents()
        assert widget._curve.curve.glstate is not None
        assert not widget._annotation_items
        assert widget.threshold_line not in widget.getPlotItem().items
    finally:
        widget.close()
