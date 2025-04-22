rom PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QMainWindow,
    QStatusBar,
    QToolBar,
)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emote")

        toolbar = QToolBar("Main Menu")
        toolbar.setIconSize(QSize(16,16))
        
        self.addToolBar(toolbar)

if __name__ == "__main__":
    musicapp = QApplication([])
    window = Window()
    window.show()
    musicapp.exec()

#link to icons: https://oclero.github.io/qlementine-icons/
