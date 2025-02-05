#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys

class My_App(QtWidgets.QMainWindow):
    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)
        
        # Fixed: Consistent 4-space indentation
        self.browse_button.clicked.connect(self.SLOT_browse_button)
    
    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]
            
            # Fixed: Consistent 4-space indentation
            pixmap = QtGui.QPixmap(self.template_path)
            self.template_label.setPixmap(pixmap)
            print("Loaded template image file: " + self.template_path)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())