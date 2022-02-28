from qtpy import QtWidgets, QtCore
import numpy as np
from typing import List, Optional


class Table(QtWidgets.QDialog):

    def __init__(self,
                 data: Optional[List] = None,
                 model=None,
                 as_dialog: bool = True,
                 **kwargs):
        if data is None:
            data = []

        super().__init__(**kwargs)
        self.title = 'Edit song definitions'

        self.data = data
        self.model = model
        self.as_dialog = as_dialog
        self.cancelled = False
        self.initUI(self.as_dialog)

    def initUI(self, as_dialog: bool = True):
        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)

        self.createTable()
        self.createButtons()

        # Add box layout, add table to box layout and add box layout to widget
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.table)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.delete_button)
        self.button_layout.addWidget(self.load_button)
        self.button_layout.addWidget(self.save_button)
        self.layout.addLayout(self.button_layout)

        if as_dialog:
            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok
                | QtWidgets.QDialogButtonBox.Cancel, QtCore.Qt.Horizontal,
                self)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            self.layout.addWidget(buttons)

        self.setLayout(self.layout)

    def createTable(self):
        # Create table
        self.table = QtWidgets.QTableWidget()
        self.table.setRowCount(len(self.data))
        if len(self.data) > 0:
            self.table.setColumnCount(len(self.data[0]))
        else:
            self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Name', 'Category'])

        for row_index, row_data in enumerate(self.data):
            self._make_row(row_index, row_data, editable_categories=False)
        self.table.move(0, 0)

    def _make_row(self,
                  row_index: Optional[int] = None,
                  row_data: Optional[List[str]] = None,
                  row_items: Optional[List[str]] = None,
                  editable_categories: bool = True):
        """[summary]

        Args:
            row_index ([type], optional): If None or omitted, will add row at end of table.
                                          Defaults to None.
            row_data (list, optional): [description]. Defaults to ['', 'segment'].
            row_items (list, optional): Dropdown items for second columns. Defaults to ["segment", "event"].
        """
        if row_index is None:
            row_index = self.table.rowCount()
            self.table.insertRow(row_index)
        if row_data is None:
            row_data = ['', 'segment']
        if row_items is None:
            row_items = ['segment', 'event']
        item = QtWidgets.QTableWidgetItem(row_data[0])
        item.original_text = row_data[0]
        self.table.setItem(row_index, 0, item)

        if editable_categories:
            cb = QtWidgets.QComboBox()
            cb.addItems(row_items)
            if row_data[1] in row_items:
                cb.setCurrentText(row_data[1])
                cb.original_text = row_data[1]
            else:
                cb.original_text = ''

            self.table.setCellWidget(row_index, 1, cb)
        else:
            item = QtWidgets.QTableWidgetItem(row_data[1])
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled
                          & ~QtCore.Qt.ItemIsEditable)
            item.original_text = row_data[1]
            self.table.setItem(row_index, 1, item)

    def get_cell_data(self, row: int, col: int):
        if self.table.item(row, col) is not None:
            return [
                self.table.item(row, col).text(),
                self.table.item(row, col).original_text
            ]
        elif self.table.cellWidget(row, col) is not None:
            return [
                self.table.cellWidget(row, col).currentText(),
                self.table.cellWidget(row, col).original_text
            ]
        else:
            return None

    def get_table_data(self):
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        data = []
        for row in range(rows):
            row_data = []
            for col in range(cols):
                row_data.append(self.get_cell_data(row, col))
            data.append(row_data)
        return data

    def createButtons(self):
        self.add_button = QtWidgets.QPushButton('New', self)
        self.add_button.clicked.connect(self.add_event)

        self.delete_button = QtWidgets.QPushButton('Delete', self)
        self.delete_button.clicked.connect(self.delete_event)

        self.load_button = QtWidgets.QPushButton(self.tr("&Load"), self)
        self.load_button.clicked.connect(self.load)

        self.save_button = QtWidgets.QPushButton(self.tr("&Save"), self)
        self.save_button.clicked.connect(self.save)

    @QtCore.Slot()
    def cancel(self):
        self.cancelled = True

    @QtCore.Slot()
    def add_event(self):
        self.data.append(['', 'segment'])
        self._make_row()

    @QtCore.Slot()
    def delete_event(self):
        del self.data[self.table.currentRow()]
        self.table.removeRow(self.table.currentRow())

    @QtCore.Slot()
    def load(self):
        filename = self.model._get_filename_from_ds(suffix="_definitions.csv")
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption="Select definitions file",
            dir=filename,
            filter=self.tr("Any File (*_definitions.csv)"))

        if len(filename):
            # load data
            data = np.loadtxt(filename, dtype=str, delimiter=",")
            # delete existing data
            self.data = []
            # clear table
            while self.table.rowCount() > 0:
                self.table.removeRow(self.table.currentRow())
            # replace with loaded data
            for d in data:
                self.data.append(d)
                self._make_row(row_data=d)

    @QtCore.Slot()
    def save(self):
        savefilename = self.model._get_filename_from_ds(
            suffix="_definitions.csv")

        savefilename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            caption="Select file for saving",
            dir=savefilename,
            filter=self.tr("Any File (*_definitions.csv)"))

        if len(savefilename):
            # sanitize data
            data = self.get_table_data()
            data = [[d[0][0], d[1][0]] for d in data]
            # save to csv
            np.savetxt(savefilename, data, delimiter=",", fmt="%s")
