from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget
from PyQt5.QtGui import QImage
from ImgProc import PP
from readTabWidget import ReadTabW
import sys
import cv2
import logging
import time


class MyWindow(QMainWindow):
    def __init__(self, pp):
        super().__init__()
        self.setWindowTitle("Procesiranje slike")
        self.pp = pp
        self.filepath = None
        self.init_ui()

    def init_ui(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setScaledContents(True)
        self.label.setMinimumSize(250, 250)

        outer_layout = QVBoxLayout()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(outer_layout)

        upper_layout = QVBoxLayout()
        bottom_layout = QVBoxLayout()

        bottom_layout.addWidget(self.label)

        outer_layout.addLayout(upper_layout)
        outer_layout.addLayout(bottom_layout)

        self.tab = ReadTabW.getTab(self.clicked)
        upper_layout.addWidget(self.tab)

        push_button = QtWidgets.QPushButton()
        push_button.setText("apply transformation")
        push_button.clicked.connect(self.pp.change_orig)
        upper_layout.addWidget(push_button)

        menubar = self.menuBar()
        self.menu_file = QtWidgets.QMenu(menubar)
        self.menu_file.setTitle("File")

        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)

        self.actionopen = QtWidgets.QAction(self)
        self.actionsave = QtWidgets.QAction(self)

        self.actionopen.setText("open")
        self.actionopen.setShortcut("Ctrl+o")
        self.actionopen.triggered.connect(self.open_image)
        self.actionsave.setText("save")
        self.actionsave.setShortcut("Ctrl+s")
        self.actionsave.triggered.connect(self.save_image)

        self.menu_file.addAction(self.actionopen)
        self.menu_file.addAction(self.actionsave)

        menubar.addAction(self.menu_file.menuAction())

    def set_img(self, image):
        if image is None:
            return
        frame = image[...,::-1].copy()
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def open_image(self, default=None):
        if default:
            image_path = default
        else:
            image_path, _ = QFileDialog.getOpenFileName()
        if not image_path:
            return
        self.pp.read_img(image_path)
        pixmap = QtGui.QPixmap(image_path)
        self.label.setPixmap(pixmap)

    def save_image(self):
        if self.pp.orig_img is None:
            return
        image_path, ext = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png)")
        if image_path:
            image_path = image_path + ext[-5:-1]
            cv2.imwrite(image_path, self.pp.orig_img)
            logging.debug(f'Image saved as: {image_path}')

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == 99:
            self.adjustSize()

    def clicked(self):
        if self.label.pixmap():
            trans = self.tab.tabText(self.tab.currentIndex())
            pars = self.tab.widget(self.tab.currentIndex()).give_vals()
            logging.debug(f"click - Transformation: {trans} \npassed parameters: {pars}")
            start_time = time.time()
            img = self.pp.transform(trans, **pars)
            logging.debug(f"done in: {round(time.time() - start_time, 5)} s")
            self.set_img(img)


def win_with_image(pp, image_path):
    w = MyWindow(pp)
    w.filepath = image_path
    w.open_image(w.filepath)
    return w


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    pp = PP()
    app = QApplication(sys.argv)
    image = "driver.jpg"
    win = win_with_image(pp, image)
    win.show()
    sys.exit(app.exec())