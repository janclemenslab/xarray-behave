try:
    import PySide2  # this will force pyqtgraph to use PySide instead of PyQt4/5
except ImportError:
    pass
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

if not hasattr(QtCore, 'Slot'):
    QtCore.Slot = QtCore.pyqtSlot


class Table(QtGui.QDialog):

    def __init__(self, data=[], **kwargs):
        super().__init__(**kwargs)
        self.title = 'Edit events'
        self.left = 0
        self.top = 0
        self.width = 300
        self.height = 200
        self.data = data
        self.cancelled = False
        self.initUI()

    def initUI(self):
        # self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)

        self.createTable()
        self.createButtons()

        # Add box layout, add table to box layout and add box layout to widget
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.table)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.delete_button)
        self.layout.addLayout(self.button_layout)

        self.button_layout = QtWidgets.QHBoxLayout()
        buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)

        self.setLayout(self.layout)

        # Show widget
        self.show()

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
        self.table.move(0,0)

    def _make_row(self, row_index=None,
                  row_data=['', 'segment'],
                  row_items=["segment", "event"],
                  editable_categories=True):
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
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled & ~QtCore.Qt.ItemIsEditable)
            item.original_text = row_data[1]
            self.table.setItem(row_index, 1, item)

    def get_cell_data(self, row: int, col: int):
        if self.table.item(row, col) is not None:
            return [self.table.item(row, col).text(), self.table.item(row, col).original_text]
        elif self.table.cellWidget(row, col) is not None:
            return [self.table.cellWidget(row, col).currentText(), self.table.cellWidget(row, col).original_text]
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
        self.add_button = QtWidgets.QPushButton('Add event', self)
        self.add_button.clicked.connect(self.add_event)

        self.delete_button = QtWidgets.QPushButton('Delete event', self)
        self.delete_button.clicked.connect(self.delete_event)

    @QtCore.Slot()
    def cancel(self):
        self.cancelled = True

    @QtCore.Slot()
    def add_event(self):
        self.data.append(['', 'segment'])
        self._make_row()

    @QtCore.Slot()
    def delete_event(self):
        # del self.data[self.table.currentRow()]
        self.table.removeRow(self.table.currentRow())
