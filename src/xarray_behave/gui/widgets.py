from typing import Optional, List, Dict
from qtpy import QtCore, QtWidgets


class ZarrOverwriteWarning(QtWidgets.QMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setIcon(QtWidgets.QMessageBox.Warning)
        self.setText("Attempting to overwrite existing zarr file.")
        self.setInformativeText(
            "This can corrupt the file and lead to data loss. \
                                 ABORT unless you know what you're doing\
                                 or save to a file with a different name."
        )
        self.setStandardButtons(QtWidgets.QMessageBox.Ignore | QtWidgets.QMessageBox.Abort)
        self.setDefaultButton(QtWidgets.QMessageBox.Abort)
        self.setEscapeButton(QtWidgets.QMessageBox.Abort)


class NoEventsRegisteredWarning(QtWidgets.QMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setIcon(QtWidgets.QMessageBox.Warning)
        self.setText("No song types added")
        self.setInformativeText("To annotate song, you first need to add song types.")
        self.setStandardButtons(QtWidgets.QMessageBox.Ignore)
        self.button = self.addButton(self.tr("Add song types"), QtWidgets.QMessageBox.ActionRole)
        self.setDefaultButton(QtWidgets.QMessageBox.Ignore)
        self.setEscapeButton(QtWidgets.QMessageBox.Ignore)


class ChkBxFileDialog(QtWidgets.QFileDialog):
    def __init__(self, caption: str = "", checkbox_titles: Optional[List[str]] = None, filter: str = "*", directory: str = ""):
        super().__init__(caption=caption, filter=filter, directory=directory)

        self.setSupportedSchemes(["file"])
        self.setOption(QtWidgets.QFileDialog.DontUseNativeDialog)
        self.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        self.selectNameFilter(filter)
        self.selectFile(directory)
        self.checkboxes = []

        if checkbox_titles is None:
            checkbox_titles = [""]
        self.checkbox_titles = checkbox_titles

        for title in self.checkbox_titles:
            self.checkboxes.append(QtWidgets.QCheckBox(title))
            self.layout().addWidget(self.checkboxes[-1])

    def checked(self, name: str) -> bool:
        """_summary_

        Args:
            name (str): _description_

        Returns:
            bool: _description_
        """
        cnt = self.checkbox_titles.index(name)
        return self.checkboxes[cnt].checkState() == QtCore.Qt.CheckState.Checked

    def set_checked(self, name: str, checked: bool):
        """_summary_

        Args:
            name (str): _description_
            checked (bool): _description_
        """
        cnt = self.checkbox_titles.index(name)
        if checked:
            self.checkboxes[cnt].setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            self.checkboxes[cnt].setCheckState(QtCore.Qt.CheckState.Unchecked)

    def get_checked_state(self) -> Dict[str, bool]:
        """_summary_

        Returns:
            Dict[str, bool]: _description_
        """
        states = {}
        for cnt, title in enumerate(self.checkbox_titles):
            states[title] = self.checkboxes[cnt].checkState() == QtCore.Qt.CheckState.Checked
        return states
