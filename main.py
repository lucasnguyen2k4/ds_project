from vnmap.gui import MapApp
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
main_window = MapApp()
main_window.show()
sys.exit(app.exec_())