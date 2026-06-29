from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ..annot import Events
from .style_profile import TIMELINE_BACKGROUND, TIMELINE_GRID, TIMELINE_PLAYHEAD, TEXT_PRIMARY, TEXT_MUTED
from .view_dialog import DoubleRangeSliderControl

EVENT_COLOR_PALETTE: tuple[tuple[str, str], ...] = (
    ("Blue", "#35b7ff"),
    ("Gold", "#ffd166"),
    ("Gray", "#8d99ae"),
    ("Coral", "#ff7f50"),
    ("Teal", "#2dd4bf"),
    ("Purple", "#9b5de5"),
    ("Green", "#42c68d"),
    ("Red", "#ff6a74"),
)
Y_AXIS_WIDTH = 72
WAVEFORM_OVERVIEW_MIN_SECONDS = 4.0


def _dialog_is_open(dialog) -> bool:
    if dialog is None:
        return False
    if hasattr(dialog, "isVisible"):
        try:
            return bool(dialog.isVisible())
        except RuntimeError:
            return False
    return True


def _activate_dialog(dialog) -> None:
    try:
        if hasattr(dialog, "show"):
            dialog.show()
        if hasattr(dialog, "raise_"):
            dialog.raise_()
        if hasattr(dialog, "activateWindow"):
            dialog.activateWindow()
    except RuntimeError:
        return


def _delete_dialog_later(dialog) -> None:
    try:
        if hasattr(dialog, "deleteLater"):
            dialog.deleteLater()
    except RuntimeError:
        return


def _configure_y_axis_inside(axis, label: str) -> None:
    axis.setWidth(Y_AXIS_WIDTH)
    axis.setLabel(label)
    axis.setStyle(
        tickLength=-7,
        tickTextOffset=-44,
        tickTextWidth=42,
        autoExpandTextSpace=False,
        autoReduceTextSpace=False,
    )
    axis.setPen(pg.mkPen(TIMELINE_GRID, width=1))
    axis.setTextPen(pg.mkPen(TEXT_MUTED))


@dataclass(frozen=True)
class EventRecord:
    id: str
    name: str
    index: int
    start_seconds: float
    stop_seconds: float
    channel: int

    @property
    def duration_seconds(self) -> float:
        return float(self.stop_seconds - self.start_seconds)


@dataclass(frozen=True)
class EventTypePreset:
    name: str
    fixed_duration: bool = True
    duration_seconds: float = 0.0
    duration_editable: bool = False
    color_hex: str = "#35b7ff"
    visible: bool = True
    editable: bool = True

    def color_tuple(self) -> tuple[int, int, int]:
        color = QtGui.QColor(self.color_hex)
        if not color.isValid():
            color = QtGui.QColor("#35b7ff")
        return color.red(), color.green(), color.blue()

    def with_name(self, name: str) -> "EventTypePreset":
        return EventTypePreset(
            name=name,
            fixed_duration=self.fixed_duration,
            duration_seconds=self.duration_seconds,
            duration_editable=self.duration_editable,
            color_hex=self.color_hex,
            visible=self.visible,
            editable=self.editable,
        )

    def with_visibility(self, visible: bool) -> "EventTypePreset":
        return EventTypePreset(
            name=self.name,
            fixed_duration=self.fixed_duration,
            duration_seconds=self.duration_seconds,
            duration_editable=self.duration_editable,
            color_hex=self.color_hex,
            visible=bool(visible),
            editable=self.editable,
        )

    def with_editability(self, editable: bool) -> "EventTypePreset":
        return EventTypePreset(
            name=self.name,
            fixed_duration=self.fixed_duration,
            duration_seconds=self.duration_seconds,
            duration_editable=self.duration_editable,
            color_hex=self.color_hex,
            visible=self.visible,
            editable=bool(editable),
        )


@dataclass(frozen=True)
class AudioChannelSettings:
    waveform_all: bool = True
    events_all: bool = True
    playback_all: bool = False
    scale_y_all: bool = True


def color_hex_from_rgb(rgb: Iterable[int]) -> str:
    values = [int(np.clip(value, 0, 255)) for value in rgb]
    while len(values) < 3:
        values.append(0)
    return "#{:02x}{:02x}{:02x}".format(values[0], values[1], values[2])


def _record_id(name: str, index: int) -> str:
    return f"{name}\x1f{int(index)}"


def records_from_events(
    events: Events,
    start_seconds: float | None = None,
    stop_seconds: float | None = None,
    channel_filter: int | None = None,
) -> list[EventRecord]:
    records: list[EventRecord] = []
    for name in events.names:
        values = np.asarray(events[name])
        if values.size == 0:
            continue
        indices = np.arange(values.shape[0])
        if start_seconds is not None or stop_seconds is not None:
            start_bound = -np.inf if start_seconds is None else float(start_seconds)
            stop_bound = np.inf if stop_seconds is None else float(stop_seconds)
            row_starts = np.minimum(values[:, 0], values[:, 1])
            row_stops = np.maximum(values[:, 0], values[:, 1])
            keep = np.logical_and(row_stops >= start_bound, row_starts <= stop_bound)
            values = values[keep]
            indices = indices[keep]
        if channel_filter is not None:
            channels = np.full(values.shape[0], -1, dtype=int)
            if values.shape[1] > 2:
                finite_channels = np.isfinite(values[:, 2])
                channels[finite_channels] = values[finite_channels, 2].astype(int)
            keep = channels == int(channel_filter)
            values = values[keep]
            indices = indices[keep]
        for index, row in zip(indices, values):
            start, stop = sorted([float(row[0]), float(row[1])])
            if not np.isfinite(start) or not np.isfinite(stop):
                continue
            channel = int(row[2]) if row.shape[0] > 2 and np.isfinite(row[2]) else -1
            records.append(
                EventRecord(
                    id=_record_id(name, index),
                    name=name,
                    index=index,
                    start_seconds=start,
                    stop_seconds=stop,
                    channel=channel,
                )
            )
    records.sort(key=lambda record: (record.start_seconds, record.stop_seconds, record.name, record.index))
    return records


class _NumericItem(QtWidgets.QTableWidgetItem):
    def __init__(self, text: str, sort_value: object, record_id: str) -> None:
        super().__init__(text)
        self._sort_value = sort_value
        self.setData(QtCore.Qt.UserRole, record_id)

    def __lt__(self, other: QtWidgets.QTableWidgetItem) -> bool:
        if isinstance(other, _NumericItem):
            return self._sort_value < other._sort_value
        return super().__lt__(other)


class EventsTableWidget(QtWidgets.QWidget):
    selection_changed = QtCore.Signal(object)
    type_changed = QtCore.Signal(object, str)
    time_changed = QtCore.Signal(object, float, float, str)
    delete_requested = QtCore.Signal(object)

    _COL_TYPE = 0
    _COL_START = 1
    _COL_STOP = 2
    _COL_DURATION = 3
    _COL_CHANNEL = 4

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._events = Events()
        self._records_by_id: dict[str, EventRecord] = {}
        self._event_names: list[str] = []
        self._locked_event_names: set[str] = set()
        self._channel_filter: int | None = None
        self._type_combo_selection_ids: list[str] | None = None
        self._sync_enabled = True
        self._window_filter_enabled = False
        self._blocked = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setObjectName("xarrayEventsTable")
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setShowGrid(False)
        self.table.setHorizontalHeaderLabels(["Event", "Start (s)", "Stop (s)", "Duration (s)", "Channel"])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(self._COL_START, QtCore.Qt.AscendingOrder)
        self.table.itemSelectionChanged.connect(self._emit_selection)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.link_checkbox = QtWidgets.QCheckBox("link table/audio views")
        self.link_checkbox.setChecked(True)
        self.link_checkbox.toggled.connect(lambda checked: setattr(self, "_sync_enabled", bool(checked)))
        row.addWidget(self.link_checkbox)
        self.window_filter_checkbox = QtWidgets.QCheckBox("filter table to audio view")
        self.window_filter_checkbox.setChecked(False)
        self.window_filter_checkbox.toggled.connect(lambda checked: setattr(self, "_window_filter_enabled", bool(checked)))
        row.addWidget(self.window_filter_checkbox)
        row.addStretch(1)
        layout.addLayout(row)

    @property
    def sync_enabled(self) -> bool:
        return bool(self._sync_enabled)

    @property
    def window_filter_enabled(self) -> bool:
        return bool(self._window_filter_enabled)

    def set_events(
        self,
        events: Events,
        selected_ids: Iterable[str] | None = None,
        locked_event_names: Iterable[str] | None = None,
        start_seconds: float | None = None,
        stop_seconds: float | None = None,
        channel_filter: int | None = None,
    ) -> None:
        selected = set(selected_ids or self.selected_record_ids())
        self._blocked = True
        self._type_combo_selection_ids = None
        self._events = Events(events)
        self._event_names = list(self._events.names)
        self._locked_event_names = set(locked_event_names or set())
        self._channel_filter = channel_filter
        records = records_from_events(
            self._events,
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
            channel_filter=channel_filter,
        )
        self._records_by_id = {record.id: record for record in records}
        sort_state = self._sort_state()
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            self._populate_row(row, record)
        self.table.blockSignals(False)
        self._restore_sort_state(sort_state)
        self._select_ids(selected & set(self._records_by_id), emit=False)
        self._blocked = False

    def selected_records(self) -> list[EventRecord]:
        ids = self.selected_record_ids()
        return [self._records_by_id[record_id] for record_id in ids if record_id in self._records_by_id]

    def selected_record_ids(self) -> list[str]:
        ids: list[str] = []
        for index in self.table.selectionModel().selectedRows():
            record_id = self._record_id_for_row(index.row())
            if record_id is not None:
                ids.append(record_id)
        return ids

    def select_ids(self, record_ids: Iterable[str]) -> None:
        self._select_ids(set(record_ids), emit=True)

    def select_overlapping_range(self, start_seconds: float, stop_seconds: float) -> None:
        selected = {
            record.id
            for record in self._records_by_id.values()
            if not (record.stop_seconds < start_seconds or record.start_seconds > stop_seconds)
        }
        if not selected:
            after = [record for record in self._records_by_id.values() if record.start_seconds >= start_seconds]
            if after:
                selected = {min(after, key=lambda record: record.start_seconds).id}
        self._select_ids(selected, emit=True, scroll=True)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            records = [record for record in self.selected_records() if record.name not in self._locked_event_names]
            if records:
                self.delete_requested.emit(records)
                event.accept()
                return
        super().keyPressEvent(event)

    def _populate_row(self, row: int, record: EventRecord) -> None:
        locked = record.name in self._locked_event_names
        type_item = self._item(record.name, record.name.lower(), record.id)
        self.table.setItem(row, self._COL_TYPE, type_item)
        combo = QtWidgets.QComboBox(self.table)
        combo.setFrame(False)
        combo.addItems(self._event_names)
        idx = combo.findText(record.name)
        combo.setCurrentIndex(max(0, idx))
        combo.setEnabled(not locked)
        combo.setProperty("xarray_behave_record_id", record.id)
        combo.installEventFilter(self)
        combo.activated.connect(lambda _idx, rid=record.id, source=combo: self._on_type_combo(rid, source.currentText()))
        self.table.setCellWidget(row, self._COL_TYPE, combo)

        self.table.setItem(
            row,
            self._COL_START,
            self._item(f"{record.start_seconds:.6f}", record.start_seconds, record.id, editable=not locked),
        )
        self.table.setItem(
            row,
            self._COL_STOP,
            self._item(f"{record.stop_seconds:.6f}", record.stop_seconds, record.id, editable=not locked),
        )
        self.table.setItem(
            row, self._COL_DURATION, self._item(f"{record.duration_seconds:.6f}", record.duration_seconds, record.id)
        )
        self.table.setItem(row, self._COL_CHANNEL, self._item(str(record.channel), record.channel, record.id))

    def _item(self, text: str, sort_value: object, record_id: str, editable: bool = False) -> QtWidgets.QTableWidgetItem:
        item = _NumericItem(text, sort_value, record_id)
        flags = item.flags()
        if editable:
            item.setFlags(flags | QtCore.Qt.ItemIsEditable)
        else:
            item.setFlags(flags & ~QtCore.Qt.ItemIsEditable)
        if isinstance(sort_value, (int, float)):
            item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return item

    def _on_type_combo(self, record_id: str, new_name: str) -> None:
        pending_selected_ids = self._type_combo_selection_ids
        self._type_combo_selection_ids = None
        if self._blocked or not new_name:
            return
        source = self._records_by_id.get(record_id)
        selected_ids = self.selected_record_ids()
        if (
            pending_selected_ids is not None
            and record_id in pending_selected_ids
            and len(pending_selected_ids) >= len(selected_ids)
        ):
            selected_ids = pending_selected_ids
        selected = [self._records_by_id[selected_id] for selected_id in selected_ids if selected_id in self._records_by_id]
        if source is not None and source.name in self._locked_event_names:
            return
        if source is not None and record_id not in selected_ids:
            selected = [source]
        selected = [record for record in selected if record.name not in self._locked_event_names]
        if selected:
            self.type_changed.emit(selected, new_name)

    def _remember_type_combo_selection(self, record_id: str) -> None:
        selected_ids = self.selected_record_ids()
        self._type_combo_selection_ids = selected_ids if record_id in selected_ids else [record_id]

    def eventFilter(self, source, event) -> bool:
        if isinstance(source, QtWidgets.QComboBox):
            record_id = source.property("xarray_behave_record_id")
            if isinstance(record_id, str) and event.type() in (
                QtCore.QEvent.KeyPress,
                QtCore.QEvent.MouseButtonPress,
            ):
                self._remember_type_combo_selection(record_id)
        return super().eventFilter(source, event)

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._blocked or item.column() not in (self._COL_START, self._COL_STOP):
            return
        record_id = item.data(QtCore.Qt.UserRole)
        record = self._records_by_id.get(record_id)
        if record is None or record.name in self._locked_event_names:
            return
        try:
            value = float(item.text())
        except ValueError:
            self.set_events(
                self._events,
                locked_event_names=self._locked_event_names,
                channel_filter=self._channel_filter,
            )
            return
        start = value if item.column() == self._COL_START else record.start_seconds
        stop = value if item.column() == self._COL_STOP else record.stop_seconds
        start, stop = sorted([start, stop])
        changed_edge = "start" if item.column() == self._COL_START else "stop"
        self.time_changed.emit(record, start, stop, changed_edge)

    def _emit_selection(self) -> None:
        if not self._blocked:
            self.selection_changed.emit(self.selected_records())

    def _record_id_for_row(self, row: int) -> str | None:
        if row < 0 or row >= self.table.rowCount():
            return None
        for column in range(self.table.columnCount()):
            item = self.table.item(row, column)
            if item is None:
                continue
            record_id = item.data(QtCore.Qt.UserRole)
            if isinstance(record_id, str):
                return record_id
        return None

    def _select_ids(self, selected: set[str], emit: bool, scroll: bool = False) -> None:
        self.table.blockSignals(True)
        self.table.clearSelection()
        model = self.table.selectionModel()
        first_item = None
        for row in range(self.table.rowCount()):
            record_id = self._record_id_for_row(row)
            if record_id in selected:
                if model is not None:
                    index = self.table.model().index(row, 0)
                    model.select(index, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
                else:
                    self.table.selectRow(row)
                if first_item is None:
                    first_item = self.table.item(row, 0)
        self.table.blockSignals(False)
        if scroll and first_item is not None:
            self.table.scrollToItem(first_item, QtWidgets.QAbstractItemView.PositionAtTop)
        if emit:
            self._emit_selection()

    def _sort_state(self):
        header = self.table.horizontalHeader()
        return self.table.isSortingEnabled(), header.sortIndicatorSection(), header.sortIndicatorOrder()

    def _restore_sort_state(self, sort_state) -> None:
        enabled, section, order = sort_state
        self.table.setSortingEnabled(enabled)
        if enabled and section >= 0:
            self.table.sortItems(section, order)


def _color_swatch_icon(color_hex: str) -> QtGui.QIcon:
    color = QtGui.QColor(color_hex)
    if not color.isValid():
        color = QtGui.QColor("#8d99ae")
    pixmap = QtGui.QPixmap(14, 14)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setPen(QtGui.QPen(QtGui.QColor("#2c3748")))
    painter.setBrush(QtGui.QBrush(color))
    painter.drawRoundedRect(1, 1, 12, 12, 3, 3)
    painter.end()
    return QtGui.QIcon(pixmap)


def _visibility_icon(visible: bool) -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(18, 18)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    pen_color = QtGui.QColor(TEXT_PRIMARY if visible else TEXT_MUTED)
    painter.setPen(QtGui.QPen(pen_color, 1.7))
    painter.setBrush(QtCore.Qt.NoBrush)
    painter.drawEllipse(QtCore.QRectF(2.5, 5.0, 13.0, 8.0))
    if visible:
        painter.setBrush(QtGui.QBrush(pen_color))
        painter.drawEllipse(QtCore.QRectF(7.0, 7.0, 4.0, 4.0))
    else:
        painter.drawLine(QtCore.QPointF(4.0, 14.0), QtCore.QPointF(14.0, 4.0))
    painter.end()
    return QtGui.QIcon(pixmap)


def _editability_icon(editable: bool) -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(18, 18)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    pen_color = QtGui.QColor(TEXT_PRIMARY if editable else TEXT_MUTED)
    painter.setPen(QtGui.QPen(pen_color, 1.7))
    painter.setBrush(QtCore.Qt.NoBrush)
    if editable:
        painter.drawArc(QtCore.QRectF(3.5, 2.5, 8.0, 8.0), 35 * 16, 240 * 16)
    else:
        painter.drawArc(QtCore.QRectF(5.0, 2.5, 8.0, 8.0), 0, 180 * 16)
    painter.setBrush(QtGui.QBrush(pen_color))
    painter.drawRoundedRect(QtCore.QRectF(4.0, 8.0, 10.0, 7.5), 1.4, 1.4)
    painter.end()
    return QtGui.QIcon(pixmap)


def _compact_tool_button(icon: QtGui.QIcon, tooltip: str, parent=None) -> QtWidgets.QToolButton:
    button = QtWidgets.QToolButton(parent)
    button.setProperty("role", "presetIcon")
    button.setAutoRaise(True)
    button.setIcon(icon)
    button.setIconSize(QtCore.QSize(18, 18))
    button.setToolTip(tooltip)
    button.setFocusPolicy(QtCore.Qt.NoFocus)
    return button


def _settings_icon() -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(18, 18)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    color = QtGui.QColor(TEXT_PRIMARY)
    painter.setPen(QtGui.QPen(color, 1.5))
    painter.setBrush(QtCore.Qt.NoBrush)
    center = QtCore.QPointF(9.0, 9.0)
    for angle in range(0, 360, 45):
        transform = QtGui.QTransform()
        transform.translate(center.x(), center.y())
        transform.rotate(angle)
        transform.translate(-center.x(), -center.y())
        painter.setTransform(transform)
        painter.drawLine(QtCore.QPointF(9.0, 1.8), QtCore.QPointF(9.0, 4.0))
    painter.resetTransform()
    painter.drawEllipse(QtCore.QRectF(4.0, 4.0, 10.0, 10.0))
    painter.drawEllipse(QtCore.QRectF(7.0, 7.0, 4.0, 4.0))
    painter.end()
    return QtGui.QIcon(pixmap)


class AudioSettingsDialog(QtWidgets.QDialog):
    settings_changed = QtCore.Signal(object)

    def __init__(self, settings: AudioChannelSettings | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Audio Settings")
        self._settings = settings or AudioChannelSettings()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        self.waveform_all_radio, self.waveform_current_radio = self._add_scope_group(
            layout,
            "Waveform",
            "Show all channels",
            "Show selected channel",
            self._settings.waveform_all,
        )
        self.scale_y_all_radio, self.scale_y_current_radio = self._add_scope_group(
            layout,
            "Y limits",
            "Scale from all visible channels",
            "Scale from selected channel",
            self._settings.scale_y_all,
        )
        self.events_all_radio, self.events_current_radio = self._add_scope_group(
            layout,
            "Annotations",
            "Show events from all channels",
            "Show events from selected channel",
            self._settings.events_all,
        )
        self.playback_all_radio, self.playback_current_radio = self._add_scope_group(
            layout,
            "Playback",
            "Play all channels",
            "Play selected channel",
            self._settings.playback_all,
        )
        for radio in (
            self.waveform_all_radio,
            self.scale_y_all_radio,
            self.events_all_radio,
            self.playback_all_radio,
        ):
            radio.toggled.connect(self._emit_settings_changed)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_scope_group(
        self,
        layout: QtWidgets.QVBoxLayout,
        title: str,
        all_label: str,
        current_label: str,
        all_checked: bool,
    ) -> tuple[QtWidgets.QRadioButton, QtWidgets.QRadioButton]:
        group = QtWidgets.QGroupBox(title, self)
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(10, 8, 10, 8)
        group_layout.setSpacing(4)
        all_radio = QtWidgets.QRadioButton(all_label, group)
        current_radio = QtWidgets.QRadioButton(current_label, group)
        all_radio.setChecked(bool(all_checked))
        current_radio.setChecked(not bool(all_checked))
        group_layout.addWidget(all_radio)
        group_layout.addWidget(current_radio)
        layout.addWidget(group)
        return all_radio, current_radio

    def settings(self) -> AudioChannelSettings:
        return AudioChannelSettings(
            waveform_all=self.waveform_all_radio.isChecked(),
            events_all=self.events_all_radio.isChecked(),
            playback_all=self.playback_all_radio.isChecked(),
            scale_y_all=self.scale_y_all_radio.isChecked(),
        )

    def _emit_settings_changed(self, _checked: bool) -> None:
        self.settings_changed.emit(self.settings())


class EventTypePresetDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        title: str,
        preset: EventTypePreset | None = None,
        used_names: Iterable[str] | None = None,
        used_color_hexes: Iterable[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._preset = preset
        self._used_names = {name for name in (used_names or []) if name}
        if preset is not None:
            self._used_names.discard(preset.name)
        self._used_color_hexes = {str(color).strip().lower() for color in (used_color_hexes or []) if str(color).strip()}
        if preset is not None:
            self._used_color_hexes.discard(preset.color_hex.strip().lower())
        self._result: EventTypePreset | None = None

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        self.name_edit = QtWidgets.QLineEdit()
        form.addRow("Name", self.name_edit)

        self.fixed_checkbox = QtWidgets.QCheckBox("Fixed duration")
        form.addRow("Fixed", self.fixed_checkbox)

        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(0.0, 3600.0)
        self.duration_spin.setDecimals(6)
        self.duration_spin.setSingleStep(0.01)
        self.duration_spin.setSuffix(" s")
        form.addRow("Duration", self.duration_spin)

        self.duration_editable_checkbox = QtWidgets.QCheckBox("Editable after create")
        form.addRow("Editable", self.duration_editable_checkbox)

        self.color_combo = QtWidgets.QComboBox()
        for color_name, color_hex in EVENT_COLOR_PALETTE:
            self.color_combo.addItem(_color_swatch_icon(color_hex), color_name, color_hex)
        form.addRow("Color", self.color_combo)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.fixed_checkbox.toggled.connect(self._sync_fixed_controls)
        self._seed(preset)
        self.adjustSize()
        self.resize(min(360, self.sizeHint().width()), self.sizeHint().height())

    def value(self) -> EventTypePreset | None:
        return self._result

    def _seed(self, preset: EventTypePreset | None) -> None:
        if preset is None:
            self.name_edit.setText(self._next_default_name())
            self.fixed_checkbox.setChecked(True)
            self.duration_spin.setValue(0.0)
            self.duration_editable_checkbox.setChecked(False)
            self._select_first_unused_color()
        else:
            self.name_edit.setText(preset.name)
            self.fixed_checkbox.setChecked(bool(preset.fixed_duration))
            self.duration_spin.setValue(max(0.0, float(preset.duration_seconds)))
            self.duration_editable_checkbox.setChecked(bool(preset.duration_editable))
            color_index = self.color_combo.findData(preset.color_hex)
            if color_index >= 0:
                self.color_combo.setCurrentIndex(color_index)
        self._sync_fixed_controls()

    def _next_default_name(self) -> str:
        base = "new_event"
        if base not in self._used_names:
            return base
        index = 2
        while f"{base}_{index}" in self._used_names:
            index += 1
        return f"{base}_{index}"

    def _select_first_unused_color(self) -> None:
        for idx in range(self.color_combo.count()):
            color_hex = str(self.color_combo.itemData(idx) or "").lower()
            if color_hex and color_hex not in self._used_color_hexes:
                self.color_combo.setCurrentIndex(idx)
                return

    def _sync_fixed_controls(self) -> None:
        enabled = self.fixed_checkbox.isChecked()
        self.duration_spin.setEnabled(enabled)
        self.duration_editable_checkbox.setEnabled(enabled)
        if not enabled:
            self.duration_editable_checkbox.setChecked(True)

    def _on_accept(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Invalid Event", "Name is required.")
            return
        if name in self._used_names:
            QtWidgets.QMessageBox.warning(self, "Invalid Event", f"Event '{name}' already exists.")
            return
        color_hex = str(self.color_combo.currentData() or "#35b7ff")
        if not QtGui.QColor(color_hex).isValid():
            color_hex = "#35b7ff"
        fixed_duration = bool(self.fixed_checkbox.isChecked())
        self._result = EventTypePreset(
            name=name,
            fixed_duration=fixed_duration,
            duration_seconds=float(self.duration_spin.value()) if fixed_duration else 0.0,
            duration_editable=bool(self.duration_editable_checkbox.isChecked()) if fixed_duration else True,
            color_hex=color_hex,
            visible=self._preset.visible if self._preset is not None else True,
            editable=self._preset.editable if self._preset is not None else True,
        )
        self.accept()


class ChannelSelectorPanel(QtWidgets.QWidget):
    channel_changed = QtCore.Signal()
    settings_requested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("channelPanel")
        self.setMinimumWidth(220)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        self.title_label = QtWidgets.QLabel("Audio")
        self.title_label.setProperty("role", "inspectorTitle")
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.settings_button = _compact_tool_button(_settings_icon(), "Audio settings", self)
        self.settings_button.setObjectName("audioSettingsButton")
        self.settings_button.setProperty("role", "presetGlobal")
        self.settings_button.clicked.connect(self.settings_requested.emit)
        header.addWidget(self.settings_button)
        layout.addLayout(header)

        self.channel_combo = QtWidgets.QComboBox(self)
        self.channel_combo.setObjectName("channelSelector")
        self.channel_combo.currentIndexChanged.connect(lambda _index: self.channel_changed.emit())
        layout.addWidget(self.channel_combo)

    def set_channels(self, labels: Iterable[str]) -> None:
        current = self.channel_combo.currentText()
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems(list(labels))
        index = self.channel_combo.findText(current)
        self.channel_combo.setCurrentIndex(max(0, index))
        self.channel_combo.setEnabled(self.channel_combo.count() > 1)
        self.channel_combo.blockSignals(False)


class ThresholdingPanel(QtWidgets.QWidget):
    threshold_changed = QtCore.Signal(float)
    envelope_std_changed = QtCore.Signal(float)
    min_distance_changed = QtCore.Signal(float)
    duration_filter_changed = QtCore.Signal(bool)
    duration_range_changed = QtCore.Signal(object)
    bandpass_filter_changed = QtCore.Signal(bool)
    bandpass_range_changed = QtCore.Signal(object)
    generate_requested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("thresholdPanel")
        self.setMinimumWidth(220)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        title = QtWidgets.QLabel("Thresholding")
        title.setProperty("role", "inspectorTitle")
        layout.addWidget(title)

        self.threshold_spin = self._double_spin(decimals=6, minimum=0.0, maximum=1.0e12, step=0.001)
        self.threshold_spin.setObjectName("thresholdValue")
        self.threshold_spin.setToolTip("Envelope threshold")
        self.threshold_spin.valueChanged.connect(lambda value: self.threshold_changed.emit(float(value)))
        layout.addLayout(self._labeled_row("Threshold", self.threshold_spin))

        self.envelope_std_spin = self._double_spin(decimals=4, minimum=0.0, maximum=1.0, step=0.001)
        self.envelope_std_spin.setObjectName("thresholdEnvelopeStd")
        self.envelope_std_spin.setToolTip("Envelope smoothing window in seconds")
        self.envelope_std_spin.valueChanged.connect(lambda value: self.envelope_std_changed.emit(float(value)))
        layout.addLayout(self._labeled_row("Envelope", self.envelope_std_spin))

        self.min_distance_spin = self._double_spin(decimals=4, minimum=0.0, maximum=100.0, step=0.001)
        self.min_distance_spin.setObjectName("thresholdMinDistance")
        self.min_distance_spin.setToolTip("Minimum distance between proposed events in seconds")
        self.min_distance_spin.valueChanged.connect(lambda value: self.min_distance_changed.emit(float(value)))
        layout.addLayout(self._labeled_row("Min gap", self.min_distance_spin))

        self.duration_checkbox = QtWidgets.QCheckBox("Duration filter")
        self.duration_checkbox.setToolTip("Create interval proposals and keep only durations in this range")
        self.duration_checkbox.toggled.connect(self._on_duration_filter_toggled)
        layout.addWidget(self.duration_checkbox)

        self.duration_range = self._range_slider(maximum=1.0, decimals=4, step=0.001)
        self.duration_range.setObjectName("thresholdDurationRange")
        self.duration_range.valueChanged.connect(lambda value: self.duration_range_changed.emit(value))
        layout.addLayout(self._labeled_row("Duration", self.duration_range))

        self.bandpass_checkbox = QtWidgets.QCheckBox("Band-pass filter")
        self.bandpass_checkbox.setToolTip("Filter audio before envelope computation")
        self.bandpass_checkbox.toggled.connect(self._on_bandpass_filter_toggled)
        layout.addWidget(self.bandpass_checkbox)

        self.bandpass_range = self._range_slider(maximum=1.0, decimals=1, step=10.0)
        self.bandpass_range.setObjectName("thresholdBandpassRange")
        self.bandpass_range.valueChanged.connect(lambda value: self.bandpass_range_changed.emit(value))
        layout.addLayout(self._labeled_row("Cutoffs", self.bandpass_range))

        self.generate_button = QtWidgets.QPushButton("Generate")
        self.generate_button.setToolTip("Generate proposals for the active event type")
        self.generate_button.clicked.connect(self.generate_requested.emit)
        layout.addWidget(self.generate_button)
        self._sync_optional_controls()

    def set_values(
        self,
        *,
        threshold: float,
        envelope_std: float,
        min_distance: float,
        duration_enabled: bool,
        duration_range: tuple[float, float],
        bandpass_enabled: bool,
        bandpass_range: tuple[float, float],
    ) -> None:
        self._set_spin_value(self.threshold_spin, threshold)
        self._set_spin_value(self.envelope_std_spin, envelope_std)
        self._set_spin_value(self.min_distance_spin, min_distance)
        self._set_checkbox_value(self.duration_checkbox, duration_enabled)
        self._set_range_value(self.duration_range, duration_range)
        self._set_checkbox_value(self.bandpass_checkbox, bandpass_enabled)
        self._set_range_value(self.bandpass_range, bandpass_range)
        self._sync_optional_controls()

    def set_threshold(self, threshold: float) -> None:
        self._set_spin_value(self.threshold_spin, threshold)

    def set_limits(self, *, duration_max: float, frequency_max: float) -> None:
        self._set_range_limit(self.duration_range, max(0.001, float(duration_max)))
        self._set_range_limit(self.bandpass_range, max(1.0, float(frequency_max)))

    def _double_spin(self, *, decimals: int, minimum: float, maximum: float, step: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox(self)
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        return spin

    def _range_slider(self, *, maximum: float, decimals: int, step: float) -> DoubleRangeSliderControl:
        slider = DoubleRangeSliderControl(self)
        slider.setRange(0.0, maximum)
        slider.setDecimals(decimals)
        slider.setSingleStep(step)
        slider.setValue([0.0, maximum])
        return slider

    def _labeled_row(self, label: str, widget: QtWidgets.QWidget) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        text = QtWidgets.QLabel(label)
        text.setProperty("role", "muted")
        row.addWidget(text)
        row.addWidget(widget, 1)
        return row

    def _set_spin_value(self, spin: QtWidgets.QDoubleSpinBox, value: float) -> None:
        spin.blockSignals(True)
        spin.setValue(float(value))
        spin.blockSignals(False)

    def _set_checkbox_value(self, checkbox: QtWidgets.QCheckBox, checked: bool) -> None:
        checkbox.blockSignals(True)
        checkbox.setChecked(bool(checked))
        checkbox.blockSignals(False)

    def _set_range_value(self, slider: DoubleRangeSliderControl, value: tuple[float, float]) -> None:
        slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(False)

    def _set_range_limit(self, slider: DoubleRangeSliderControl, maximum: float) -> None:
        value = slider.value()
        slider.blockSignals(True)
        slider.setRange(0.0, maximum)
        slider.setValue([min(value[0], maximum), min(value[1], maximum)])
        slider.blockSignals(False)

    def _on_duration_filter_toggled(self, checked: bool) -> None:
        self._sync_optional_controls()
        self.duration_filter_changed.emit(bool(checked))

    def _on_bandpass_filter_toggled(self, checked: bool) -> None:
        self._sync_optional_controls()
        self.bandpass_filter_changed.emit(bool(checked))

    def _sync_optional_controls(self) -> None:
        self.duration_range.setEnabled(self.duration_checkbox.isChecked())
        self.bandpass_range.setEnabled(self.bandpass_checkbox.isChecked())


class _PresetRowWidget(QtWidgets.QWidget):
    selection_requested = QtCore.Signal(str)
    edit_requested = QtCore.Signal(str)
    visibility_changed = QtCore.Signal(str, bool)
    editability_changed = QtCore.Signal(str, bool)

    def __init__(self, preset: EventTypePreset, detail_text: str, parent=None) -> None:
        super().__init__(parent)
        self._preset_name = preset.name

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(5)

        self.visibility_button = _compact_tool_button(_visibility_icon(preset.visible), "", self)
        self.visibility_button.setCheckable(True)
        self.visibility_button.setChecked(bool(preset.visible))
        self.visibility_button.clicked.connect(self._on_visibility_clicked)
        layout.addWidget(self.visibility_button)

        self.editability_button = _compact_tool_button(_editability_icon(preset.editable), "", self)
        self.editability_button.setCheckable(True)
        self.editability_button.setChecked(bool(preset.editable))
        self.editability_button.clicked.connect(self._on_editability_clicked)
        layout.addWidget(self.editability_button)

        swatch = QtWidgets.QLabel(self)
        swatch.setPixmap(_color_swatch_icon(preset.color_hex).pixmap(14, 14))
        layout.addWidget(swatch)

        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(0)
        self.name_label = QtWidgets.QLabel(preset.name)
        self.name_label.setProperty("role", "presetName")
        self.name_label.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.detail_label = QtWidgets.QLabel(detail_text)
        self.detail_label.setProperty("role", "muted")
        self.detail_label.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        text_layout.addWidget(self.name_label)
        text_layout.addWidget(self.detail_label)
        layout.addLayout(text_layout, 1)

        self._sync_visibility_button()
        self._sync_editability_button()

    def mousePressEvent(self, event) -> None:
        self.selection_requested.emit(self._preset_name)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.edit_requested.emit(self._preset_name)
        super().mouseDoubleClickEvent(event)

    def _on_visibility_clicked(self, checked: bool) -> None:
        self._sync_visibility_button()
        self.visibility_changed.emit(self._preset_name, bool(checked))

    def _on_editability_clicked(self, checked: bool) -> None:
        self._sync_editability_button()
        self.editability_changed.emit(self._preset_name, bool(checked))

    def _sync_visibility_button(self) -> None:
        visible = bool(self.visibility_button.isChecked())
        self.visibility_button.setIcon(_visibility_icon(visible))
        self.visibility_button.setToolTip(f"Hide {self._preset_name}" if visible else f"Show {self._preset_name}")

    def _sync_editability_button(self) -> None:
        editable = bool(self.editability_button.isChecked())
        self.editability_button.setIcon(_editability_icon(editable))
        self.editability_button.setToolTip(f"Lock {self._preset_name}" if editable else f"Unlock {self._preset_name}")


class EventPresetPanel(QtWidgets.QWidget):
    selection_changed = QtCore.Signal(str)
    create_requested = QtCore.Signal()
    edit_requested = QtCore.Signal(str)
    delete_requested = QtCore.Signal(str)
    visibility_changed = QtCore.Signal(str, bool)
    editability_changed = QtCore.Signal(str, bool)
    visibility_all_changed = QtCore.Signal(bool)
    editability_all_changed = QtCore.Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("presetPanel")
        self.setMinimumWidth(220)
        self._presets: list[EventTypePreset] = []
        self._blocked = False
        self._visibility_all_target = True
        self._editability_all_target = True

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        self.title_label = QtWidgets.QLabel("Annotations")
        self.title_label.setProperty("role", "inspectorTitle")
        header.addWidget(self.title_label)
        header.addStretch(1)

        self.visibility_all_button = _compact_tool_button(_visibility_icon(True), "Hide all", self)
        self.visibility_all_button.setProperty("role", "presetGlobal")
        self.visibility_all_button.clicked.connect(self._emit_visibility_all)
        header.addWidget(self.visibility_all_button)

        self.editability_all_button = _compact_tool_button(_editability_icon(True), "Lock all", self)
        self.editability_all_button.setProperty("role", "presetGlobal")
        self.editability_all_button.clicked.connect(self._emit_editability_all)
        header.addWidget(self.editability_all_button)
        layout.addLayout(header)

        self.list_widget = QtWidgets.QListWidget(self)
        self.list_widget.setObjectName("presetList")
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_widget.currentItemChanged.connect(self._on_current_item_changed)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget, 1)

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        self.new_button = QtWidgets.QPushButton("New")
        self.edit_button = QtWidgets.QPushButton("Edit")
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.new_button.clicked.connect(self.create_requested.emit)
        self.edit_button.clicked.connect(self._emit_edit)
        self.delete_button.clicked.connect(self._emit_delete)
        controls.addWidget(self.new_button)
        controls.addWidget(self.edit_button)
        controls.addWidget(self.delete_button)
        controls.addStretch(1)
        layout.addLayout(controls)

    def set_presets(self, presets: Iterable[EventTypePreset], selected_name: str | None = None) -> None:
        self._blocked = True
        self._presets = list(presets)
        self.list_widget.clear()
        if not self._presets:
            placeholder = QtWidgets.QListWidgetItem("No event presets")
            placeholder.setFlags(QtCore.Qt.NoItemFlags)
            self.list_widget.addItem(placeholder)
            self.edit_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            self._sync_global_buttons()
            self._blocked = False
            return
        selected_row = 0
        for row, preset in enumerate(self._presets):
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.UserRole, preset.name)
            item.setToolTip(self._format_preset_tooltip(preset))
            self.list_widget.addItem(item)
            row_widget = _PresetRowWidget(preset, self._format_preset_detail(preset), self.list_widget)
            row_widget.selection_requested.connect(self._select_name)
            row_widget.edit_requested.connect(self.edit_requested.emit)
            row_widget.visibility_changed.connect(self.visibility_changed.emit)
            row_widget.editability_changed.connect(self.editability_changed.emit)
            item.setSizeHint(row_widget.sizeHint())
            self.list_widget.setItemWidget(item, row_widget)
            if preset.name == selected_name:
                selected_row = row
        self.list_widget.setCurrentRow(selected_row)
        self.edit_button.setEnabled(True)
        self.delete_button.setEnabled(True)
        self._sync_global_buttons()
        self._blocked = False

    def set_current_name(self, name: str | None) -> None:
        if name is None:
            return
        self._blocked = True
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is not None and item.data(QtCore.Qt.UserRole) == name:
                self.list_widget.setCurrentRow(row)
                break
        self._blocked = False

    def current_name(self) -> str | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        name = item.data(QtCore.Qt.UserRole)
        return name if isinstance(name, str) and name else None

    def _on_current_item_changed(self, current, _previous) -> None:
        if self._blocked:
            return
        name = current.data(QtCore.Qt.UserRole) if current is not None else None
        if isinstance(name, str) and name:
            self.selection_changed.emit(name)

    def _on_item_double_clicked(self, item) -> None:
        name = item.data(QtCore.Qt.UserRole) if item is not None else None
        if isinstance(name, str) and name:
            self.edit_requested.emit(name)

    def _emit_edit(self) -> None:
        name = self.current_name()
        if name:
            self.edit_requested.emit(name)

    def _emit_delete(self) -> None:
        name = self.current_name()
        if name:
            self.delete_requested.emit(name)

    def _select_name(self, name: str) -> None:
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            item_name = item.data(QtCore.Qt.UserRole) if item is not None else None
            if item_name != name:
                continue
            if self.list_widget.currentRow() == row:
                self.selection_changed.emit(name)
                return
            self.list_widget.setCurrentRow(row)
            return

    def _emit_visibility_all(self) -> None:
        self.visibility_all_changed.emit(bool(self._visibility_all_target))

    def _emit_editability_all(self) -> None:
        self.editability_all_changed.emit(bool(self._editability_all_target))

    def _sync_global_buttons(self) -> None:
        has_presets = bool(self._presets)
        all_visible = has_presets and all(preset.visible for preset in self._presets)
        all_editable = has_presets and all(preset.editable for preset in self._presets)
        self._visibility_all_target = not all_visible
        self._editability_all_target = not all_editable
        self.visibility_all_button.setEnabled(has_presets)
        self.editability_all_button.setEnabled(has_presets)
        self.visibility_all_button.setIcon(_visibility_icon(self._visibility_all_target))
        self.editability_all_button.setIcon(_editability_icon(self._editability_all_target))
        self.visibility_all_button.setToolTip("Show all" if self._visibility_all_target else "Hide all")
        self.editability_all_button.setToolTip("Unlock all" if self._editability_all_target else "Lock all")

    def _format_preset_detail(self, preset: EventTypePreset) -> str:
        if not preset.fixed_duration:
            return "free"
        edit_text = "duration editable" if preset.duration_editable else "duration locked"
        return f"fixed {preset.duration_seconds:g}s, {edit_text}"

    def _format_preset_tooltip(self, preset: EventTypePreset) -> str:
        visibility_text = "visible" if preset.visible else "hidden"
        editability_text = "editable" if preset.editable else "locked"
        if not preset.fixed_duration:
            return f"{preset.name}: free-duration event, {visibility_text}, {editability_text}"
        edit_text = "duration editable after creation" if preset.duration_editable else "duration locked after creation"
        return f"{preset.name}: fixed duration {preset.duration_seconds:g}s, {edit_text}, {visibility_text}, {editability_text}"


class WaveformSettingsDialog(QtWidgets.QDialog):
    def __init__(self, waveform: "WaveformPane", parent=None) -> None:
        super().__init__(parent)
        self.waveform = waveform
        self.setWindowTitle("Waveform display settings")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        color_row = QtWidgets.QHBoxLayout()
        color_label = QtWidgets.QLabel("Color")
        color_label.setFixedWidth(120)
        self.color_combo = QtWidgets.QComboBox(self)
        for name, color in EVENT_COLOR_PALETTE:
            self.color_combo.addItem(_color_swatch_icon(color), name, color)
        current_color = self.waveform.waveform_color
        color_index = self.color_combo.findData(current_color)
        if color_index < 0:
            self.color_combo.addItem(_color_swatch_icon(current_color), current_color, current_color)
            color_index = self.color_combo.count() - 1
        self.color_combo.setCurrentIndex(color_index)
        self.color_combo.currentIndexChanged.connect(self._on_color_changed)
        color_row.addWidget(color_label)
        color_row.addWidget(self.color_combo, 1)
        layout.addLayout(color_row)

        self.auto_limits_checkbox = QtWidgets.QCheckBox("Auto y limits", self)
        self.auto_limits_checkbox.setChecked(self.waveform.waveform_y_limits is None)
        self.auto_limits_checkbox.stateChanged.connect(self._on_limits_changed)
        layout.addWidget(self.auto_limits_checkbox)

        limits = self.waveform.waveform_y_limits
        if limits is None:
            limits = tuple(float(value) for value in self.waveform.viewRange()[1])
        self.lower_spin = self._limit_spin(limits[0])
        self.upper_spin = self._limit_spin(limits[1])
        self.lower_spin.valueChanged.connect(lambda _value: self._on_limits_changed())
        self.upper_spin.valueChanged.connect(lambda _value: self._on_limits_changed())

        limits_layout = QtWidgets.QGridLayout()
        limits_layout.addWidget(QtWidgets.QLabel("Lower"), 0, 0)
        limits_layout.addWidget(self.lower_spin, 0, 1)
        limits_layout.addWidget(QtWidgets.QLabel("Upper"), 1, 0)
        limits_layout.addWidget(self.upper_spin, 1, 1)
        layout.addLayout(limits_layout)
        self._sync_limit_controls()

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _limit_spin(self, value: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox(self)
        spin.setRange(-1.0e12, 1.0e12)
        spin.setDecimals(6)
        spin.setValue(float(value))
        return spin

    def _on_color_changed(self) -> None:
        self.waveform.set_waveform_color(str(self.color_combo.currentData()))

    def _on_limits_changed(self) -> None:
        self._sync_limit_controls()
        if self.auto_limits_checkbox.isChecked():
            self.waveform.set_waveform_y_limits(None)
            return
        self.waveform.set_waveform_y_limits((self.lower_spin.value(), self.upper_spin.value()))

    def _sync_limit_controls(self) -> None:
        enabled = not self.auto_limits_checkbox.isChecked()
        self.lower_spin.setEnabled(enabled)
        self.upper_spin.setEnabled(enabled)


class WaveformPane(pg.PlotWidget):
    threshold_changed = QtCore.Signal(float)

    def __init__(self, parent=None, callback=None, region_changed_callback=None, position_changed_callback=None) -> None:
        super().__init__(parent=parent)
        self.setMinimumHeight(60)
        self.setBackground(TIMELINE_BACKGROUND)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True, alpha=0.16)
        self.hideAxis("bottom")
        self.setDefaultPadding(0)
        _configure_y_axis_inside(self.getAxis("left"), "Waveform")
        axis_pen = pg.mkPen(TIMELINE_GRID, width=1)
        for axis_name in ("bottom",):
            axis = self.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(pg.mkPen(TEXT_MUTED))
        self.callback = callback
        self.region_changed_callback = region_changed_callback
        self.position_changed_callback = position_changed_callback
        self._waveform_color = "#36cfc9"
        self._waveform_y_limits: tuple[float, float] | None = None
        self._last_waveform_y: np.ndarray | None = None
        self._last_waveform_y_other: np.ndarray | None = None
        self._settings_dialog: WaveformSettingsDialog | None = None
        self.settings_button = _compact_tool_button(_settings_icon(), "Waveform display settings", self)
        self.settings_button.setObjectName("waveformSettingsButton")
        self.settings_button.setProperty("role", "presetGlobal")
        self.settings_button.setFixedSize(22, 22)
        self.settings_button.clicked.connect(self._open_settings_dialog)
        self._curve = self.plot(pen=pg.mkPen(self._waveform_color, width=1.6))
        self._curve.setZValue(0)
        self._other_curves: list[pg.PlotCurveItem] = []
        self._threshold_curve = pg.PlotCurveItem(pen=pg.mkPen(color=[196, 98, 98], width=1))
        self._threshold_curve.setZValue(8)
        self._threshold_enabled = False
        self._playhead = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(TIMELINE_PLAYHEAD, width=1))
        self._playhead.setZValue(20)
        self.addItem(self._playhead)
        self._annotation_items: list[pg.GraphicsObject] = []
        self.threshold_line = pg.InfiniteLine(
            movable=True,
            angle=0,
            pos=0,
            pen=pg.mkPen(color="r", width=2, alpha=0.25),
            bounds=[0, None],
            label="Threshold",
            labelOpts={"position": 0.9},
        )
        self.threshold_line.setZValue(9)
        self.threshold_line.sigPositionChangeFinished.connect(self._on_threshold_line_changed)
        self.getPlotItem().mouseClickEvent = self._click
        self._position_settings_button()

    @property
    def threshold(self) -> float:
        return float(self.threshold_line.value())

    @property
    def waveform_color(self) -> str:
        return self._waveform_color

    @property
    def waveform_y_limits(self) -> tuple[float, float] | None:
        return self._waveform_y_limits

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_settings_button()

    def _position_settings_button(self) -> None:
        if not hasattr(self, "settings_button"):
            return
        margin = 8
        left = max(margin, self.width() - self.settings_button.width() - margin)
        self.settings_button.move(left, margin)
        self.settings_button.raise_()

    def _open_settings_dialog(self) -> None:
        if _dialog_is_open(self._settings_dialog):
            _activate_dialog(self._settings_dialog)
            return
        dialog = WaveformSettingsDialog(self, parent=self.window())
        self._settings_dialog = dialog
        dialog.finished.connect(lambda _result, active_dialog=dialog: self._clear_settings_dialog(active_dialog))
        _activate_dialog(dialog)

    def _clear_settings_dialog(self, dialog) -> None:
        if self._settings_dialog is dialog:
            self._settings_dialog = None
        _delete_dialog_later(dialog)

    def set_waveform_color(self, color: str) -> None:
        qcolor = QtGui.QColor(color)
        if not qcolor.isValid():
            return
        self._waveform_color = qcolor.name()
        self._curve.setPen(pg.mkPen(self._waveform_color, width=1.6))

    def set_waveform_y_limits(self, limits: tuple[float, float] | None) -> None:
        if limits is None:
            self._waveform_y_limits = None
        else:
            lower, upper = sorted(float(value) for value in limits)
            if lower == upper:
                upper = lower + 1.0
            self._waveform_y_limits = (lower, upper)
        if self._last_waveform_y is not None:
            self._set_visible_y_range(self._last_waveform_y, self._last_waveform_y_other)

    def set_waveform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_other: np.ndarray | None = None,
        scale_y_all: bool = True,
        max_points: int = 6000,
    ) -> None:
        self._clear_other_curves()
        if x is None or y is None or len(x) == 0 or len(y) == 0:
            self._curve.setData([], [])
            return
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim > 1:
            y = y.mean(axis=1)
        self._last_waveform_y = y
        self._last_waveform_y_other = y_other if y_other is None else np.asarray(y_other, dtype=float)
        if y_other is not None:
            y_other = self._last_waveform_y_other
            if y_other.ndim == 1:
                y_other = y_other[:, None]
                self._last_waveform_y_other = y_other
            for channel in range(y_other.shape[1]):
                curve = pg.PlotCurveItem(pen=pg.mkPen("#4d5968", width=0.9))
                curve.setZValue(-5)
                self.addItem(curve)
                self._set_curve_data(curve, x, y_other[:, channel], max_points=max_points)
                self._other_curves.append(curve)
        self._set_curve_data(self._curve, x, y, max_points=max_points)
        self.setXRange(float(x[0]), float(x[-1]), padding=0)
        self._set_visible_y_range(y, y_other if scale_y_all else None)

    def set_threshold_value(self, threshold: float) -> None:
        self.threshold_line.blockSignals(True)
        self.threshold_line.setValue(max(0.0, float(threshold)))
        self.threshold_line.blockSignals(False)
        if self._threshold_enabled:
            self._include_threshold_y_range()

    def set_threshold_data(
        self,
        x: np.ndarray | None,
        envelope: np.ndarray | None,
        *,
        enabled: bool,
        threshold: float | None = None,
        max_points: int = 6000,
    ) -> None:
        self._set_threshold_enabled(bool(enabled))
        if threshold is not None:
            self.set_threshold_value(threshold)
        if not enabled or x is None or envelope is None or len(x) == 0 or len(envelope) == 0:
            self._threshold_curve.setData([], [])
            return
        x = np.asarray(x, dtype=float)
        envelope = np.asarray(envelope, dtype=float)
        if len(envelope) != len(x):
            size = min(len(x), len(envelope))
            x = x[:size]
            envelope = envelope[:size]
        self._set_curve_data(self._threshold_curve, x, envelope, max_points=max_points)
        self._include_threshold_y_range(envelope)

    def _set_threshold_enabled(self, enabled: bool) -> None:
        if enabled == self._threshold_enabled:
            return
        self._threshold_enabled = enabled
        if enabled:
            self.addItem(self._threshold_curve)
            self.addItem(self.threshold_line)
        else:
            self.removeItem(self._threshold_curve)
            self.removeItem(self.threshold_line)

    def _include_threshold_y_range(self, envelope: np.ndarray | None = None) -> None:
        if self._waveform_y_limits is not None:
            self.setYRange(*self._waveform_y_limits, padding=0)
            return
        values = [np.array([self.threshold], dtype=float)]
        if envelope is not None:
            values.append(np.ravel(envelope))
        finite = np.concatenate(values)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        ymin, ymax = self.viewRange()[1]
        ymin = min(float(ymin), float(np.min(finite)))
        ymax = max(float(ymax), float(np.max(finite)))
        span = ymax - ymin
        margin = max(span * 0.05, 1.0e-9)
        self.setYRange(ymin - margin, ymax + margin, padding=0)

    def _on_threshold_line_changed(self) -> None:
        self.threshold_changed.emit(self.threshold)

    def _clear_other_curves(self) -> None:
        for curve in self._other_curves:
            self.removeItem(curve)
        self._other_curves = []

    def _set_visible_y_range(self, y: np.ndarray, y_other: np.ndarray | None = None) -> None:
        if self._waveform_y_limits is not None:
            self.setYRange(*self._waveform_y_limits, padding=0)
            return
        values = [np.ravel(y)]
        if y_other is not None:
            values.append(np.ravel(y_other))
        finite = np.concatenate(values)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        ymin = float(np.min(finite))
        ymax = float(np.max(finite))
        span = ymax - ymin
        if span <= 0:
            margin = max(abs(ymin) * 0.1, 1.0)
        else:
            margin = span * 0.05
        self.setYRange(ymin - margin, ymax + margin, padding=0)

    def _set_curve_data(
        self,
        curve: pg.PlotCurveItem,
        x: np.ndarray,
        y: np.ndarray,
        *,
        max_points: int,
    ) -> None:
        span_seconds = float(x[-1] - x[0]) if len(x) else 0.0
        if span_seconds >= WAVEFORM_OVERVIEW_MIN_SECONDS and len(y) > max_points:
            stride = int(np.ceil(len(y) / max_points))
            usable = (len(y) // stride) * stride
            y_block = y[:usable].reshape(-1, stride)
            x_block = x[:usable:stride]
            y_min = y_block.min(axis=1)
            y_max = y_block.max(axis=1)
            x_plot = np.repeat(x_block, 2)
            y_plot = np.empty(y_min.size * 2, dtype=float)
            y_plot[0::2] = y_min
            y_plot[1::2] = y_max
            curve.setData(x_plot, y_plot, connect="pairs")
        else:
            curve.setData(x, y, connect="finite")

    def set_playhead(self, seconds: float) -> None:
        self._playhead.setPos(float(seconds))

    def clear_annotations(self) -> None:
        for item in self._annotation_items:
            self.removeItem(item)
        self._annotation_items = []

    def add_segment(self, onset, offset, region_typeindex, brush=None, pen=None, movable=True, text=None) -> None:
        region = pg.LinearRegionItem(values=(onset, offset), movable=movable, brush=brush)
        region.event_index = region_typeindex
        region.bounds = (onset, offset)
        if pen is not None:
            for line in region.lines:
                line.setPen(pen)
        if text is not None:
            pg.InfLineLabel(region.lines[1], text, position=0.95, rotateAxis=(1, 0), anchor=(1, 1))
        self.addItem(region)
        self._annotation_items.append(region)
        if movable and self.region_changed_callback is not None:
            region.sigRegionChangeFinished.connect(self.region_changed_callback)

    def add_event(self, xx, event_type, pen, movable=False, text=None) -> None:
        if not len(xx):
            return
        for x in xx:
            line = pg.InfiniteLine(pos=x, angle=90, movable=movable, pen=pen)
            line.event_index = event_type
            line.position = x
            if text is not None:
                pg.InfLineLabel(line, text, position=0.95, rotateAxis=(1, 0), anchor=(1, 1))
            self.addItem(line)
            self._annotation_items.append(line)
            if movable and self.position_changed_callback is not None:
                line.sigPositionChangeFinished.connect(self.position_changed_callback)

    def _click(self, event) -> None:
        event.accept()
        if self.callback is None:
            return
        pos = event.pos()
        seconds = self.getPlotItem().getViewBox().mapSceneToView(pos).x()
        self.callback(seconds, event.button())


class EventBarsView(pg.PlotWidget):
    event_selected = QtCore.Signal(object)
    event_changed = QtCore.Signal(object, str, float, float)
    event_created = QtCore.Signal(str, float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setMinimumHeight(70)
        self.setBackground(TIMELINE_BACKGROUND)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True, alpha=0.16)
        self.setDefaultPadding(0)
        self.setLabel("bottom", "Time", units="s")
        _configure_y_axis_inside(self.getAxis("left"), "Events")
        axis_pen = pg.mkPen(TIMELINE_GRID, width=1)
        for axis_name in ("bottom",):
            axis = self.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(pg.mkPen(TEXT_MUTED))
        self._events = Events()
        self._records: list[EventRecord] = []
        self._rows: list[str] = []
        self._row_index: dict[str, int] = {}
        self._items: list[pg.BarGraphItem] = []
        self._selected_ids: set[str] = set()
        self._locked_duration_ids: set[str] = set()
        self._locked_event_names: set[str] = set()
        self._colors: dict[str, tuple[int, int, int]] = {}
        self._drag = None
        self._playhead = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(TIMELINE_PLAYHEAD, width=1))
        self.addItem(self._playhead)

    def set_events(
        self,
        events: Events,
        colors: dict[str, tuple[int, int, int]] | None = None,
        selected_ids: Iterable[str] | None = None,
        locked_duration_ids: Iterable[str] | None = None,
        locked_event_names: Iterable[str] | None = None,
        start_seconds: float | None = None,
        stop_seconds: float | None = None,
        channel_filter: int | None = None,
    ) -> None:
        self._events = Events(events)
        self._records = records_from_events(
            self._events,
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
            channel_filter=channel_filter,
        )
        self._rows = list(self._events.names)
        self._row_index = {name: index for index, name in enumerate(self._rows)}
        self._colors = dict(colors or {})
        self._selected_ids = set(selected_ids or self._selected_ids)
        self._locked_duration_ids = set(locked_duration_ids or set())
        self._locked_event_names = set(locked_event_names or set())
        self._redraw()

    def set_selected_ids(self, selected_ids: Iterable[str]) -> None:
        self._selected_ids = set(selected_ids)
        self._redraw()

    def set_playhead(self, seconds: float) -> None:
        self._playhead.setPos(float(seconds))

    def mousePressEvent(self, event) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            super().mousePressEvent(event)
            return
        point = self._event_point(event)
        record = self._pick_record(float(point.x()), float(point.y()))
        if record is not None:
            if record.name in self._locked_event_names:
                self.event_selected.emit([record])
                event.accept()
                return
            self._drag = {
                "record": record,
                "mode": self._drag_mode(record, float(point.x())),
                "anchor": float(point.x()),
                "start": record.start_seconds,
                "stop": record.stop_seconds,
                "row": self._row_index.get(record.name, 0),
            }
            self.event_selected.emit([record])
            event.accept()
            return
        row = self._row_for_y(float(point.y()))
        if row is not None:
            if self._rows[row] in self._locked_event_names:
                event.accept()
                return
            self._drag = {"record": None, "mode": "create", "anchor": float(point.x()), "row": row}
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag is None:
            super().mouseReleaseEvent(event)
            return
        point = self._event_point(event)
        end_time = max(0.0, float(point.x()))
        row = self._row_for_y(float(point.y()))
        if row is None:
            row = int(self._drag["row"])
        name = self._rows[row]
        record = self._drag["record"]
        if record is None:
            start, stop = sorted([float(self._drag["anchor"]), end_time])
            self.event_created.emit(name, start, stop)
        else:
            mode = self._drag["mode"]
            if mode == "resize_start":
                start, stop = sorted([end_time, float(self._drag["stop"])])
                start = max(0.0, start)
                stop = max(start, stop)
            elif mode == "resize_stop":
                start, stop = sorted([float(self._drag["start"]), end_time])
                start = max(0.0, start)
                stop = max(start, stop)
            else:
                delta = end_time - float(self._drag["anchor"])
                start = max(0.0, float(self._drag["start"]) + delta)
                stop = max(start, float(self._drag["stop"]) + delta)
            self.event_changed.emit(record, name, start, stop)
        self._drag = None
        event.accept()

    def _redraw(self) -> None:
        for item in self._items:
            self.removeItem(item)
        self._items.clear()
        if self._rows:
            ticks = [[(index, name) for name, index in self._row_index.items()]]
            self.getAxis("left").setTicks(ticks)
            self.setYRange(-0.6, len(self._rows) - 0.4, padding=0)
        else:
            self.getAxis("left").setTicks([[]])
            self.setYRange(-0.5, 0.5, padding=0)
        for record in self._records:
            row = self._row_index.get(record.name)
            if row is None:
                continue
            color = self._colors.get(record.name, (100, 180, 255))
            selected = record.id in self._selected_ids
            locked = record.name in self._locked_event_names
            alpha = 210 if selected else 120
            if locked and not selected:
                alpha = 70
            pen_width = 2.2 if selected else 1.0
            start = record.start_seconds
            stop = record.stop_seconds
            if start == stop:
                width = max(0.002, self._marker_width())
                start -= width / 2
                stop += width / 2
            bar = pg.BarGraphItem(
                x0=[start],
                x1=[max(stop, start + 1e-6)],
                y=[row - 0.34],
                height=[0.68],
                pen=pg.mkPen(color=color, width=pen_width),
                brush=pg.mkBrush(*color, alpha),
            )
            self.addItem(bar)
            self._items.append(bar)

    def _pick_record(self, seconds: float, y_value: float) -> EventRecord | None:
        row = self._row_for_y(y_value)
        if row is None:
            return None
        candidates = [record for record in self._records if self._row_index.get(record.name) == row]
        tol = self._hit_tolerance()
        best = None
        best_span = None
        for record in candidates:
            start, stop = record.start_seconds, record.stop_seconds
            if start == stop:
                hit = abs(seconds - start) <= tol
                span = 0
            else:
                hit = (start <= seconds <= stop) or min(abs(seconds - start), abs(seconds - stop)) <= tol
                span = stop - start
            if hit and (best_span is None or span < best_span):
                best = record
                best_span = span
        return best

    def _row_for_y(self, y_value: float) -> int | None:
        if not self._rows:
            return None
        row = int(round(y_value))
        if row < 0 or row >= len(self._rows):
            return None
        return row

    def _hit_tolerance(self) -> float:
        x_min, x_max = self.plotItem.vb.viewRange()[0]
        return max(0.025, float(x_max - x_min) * 0.01)

    def _marker_width(self) -> float:
        return max(0.002, self._hit_tolerance() * 0.4)

    def _drag_mode(self, record: EventRecord, seconds: float) -> str:
        if record.id in self._locked_duration_ids:
            return "move"
        if record.start_seconds == record.stop_seconds:
            return "move"
        tol = self._hit_tolerance()
        if abs(seconds - record.start_seconds) <= tol:
            return "resize_start"
        if abs(seconds - record.stop_seconds) <= tol:
            return "resize_stop"
        return "move"

    def _event_point(self, event) -> QtCore.QPointF:
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        return self.plotItem.vb.mapSceneToView(self.mapToScene(pos))


class EventTimelineWidget(QtWidgets.QWidget):
    event_selected = QtCore.Signal(object)
    event_changed = QtCore.Signal(object, str, float, float)
    event_created = QtCore.Signal(str, float, float)

    def __init__(self, parent=None, *, show_waveform: bool = True) -> None:
        super().__init__(parent)
        self.setMinimumHeight(130 if show_waveform else 70)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.events = EventBarsView()
        self.waveform = WaveformPane() if show_waveform else None
        self.splitter = None
        if self.waveform is None:
            layout.addWidget(self.events)
        else:
            self.events.setXLink(self.waveform)
            self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            self.splitter.setHandleWidth(8)
            self.splitter.addWidget(self.waveform)
            self.splitter.addWidget(self.events)
            self.splitter.setSizes([80, 100])
            self.splitter.setCollapsible(0, False)
            self.splitter.setCollapsible(1, False)
            layout.addWidget(self.splitter)
        self.events.event_selected.connect(self.event_selected.emit)
        self.events.event_changed.connect(self.event_changed.emit)
        self.events.event_created.connect(self.event_created.emit)

    def set_waveform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_other: np.ndarray | None = None,
        scale_y_all: bool = True,
    ) -> None:
        if self.waveform is not None:
            self.waveform.set_waveform(x, y, y_other=y_other, scale_y_all=scale_y_all)
            return
        if x is not None and len(x):
            self.events.setXRange(float(x[0]), float(x[-1]), padding=0)

    def set_events(
        self,
        events: Events,
        colors: dict[str, tuple[int, int, int]] | None = None,
        selected_ids: Iterable[str] | None = None,
        locked_duration_ids: Iterable[str] | None = None,
        locked_event_names: Iterable[str] | None = None,
        start_seconds: float | None = None,
        stop_seconds: float | None = None,
        channel_filter: int | None = None,
    ) -> None:
        self.events.set_events(
            events,
            colors=colors,
            selected_ids=selected_ids,
            locked_duration_ids=locked_duration_ids,
            locked_event_names=locked_event_names,
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
            channel_filter=channel_filter,
        )

    def set_selected_ids(self, selected_ids: Iterable[str]) -> None:
        self.events.set_selected_ids(selected_ids)

    def set_playhead(self, seconds: float) -> None:
        if self.waveform is not None:
            self.waveform.set_playhead(seconds)
        self.events.set_playhead(seconds)
