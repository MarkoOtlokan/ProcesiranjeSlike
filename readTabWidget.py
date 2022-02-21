from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, \
    QSlider, QPushButton, QCheckBox, QLabel, QComboBox
from PyQt5.QtCore import Qt

from WidgetHelper import MyWidget


class ReadTabW:
    def getTab(slot):
        tab = QTabWidget()

        # BRIGHTNESS
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("add_brightness")
        sl1.setValue(0)
        sl1.setMinimum(-255)
        sl1.setMaximum(255)
        sl1.setSingleStep(10)

        w = MyWidget((sl1,), name="brightness", slot=slot)

        # CONTRAST
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("p")
        sl1.setValue(0)
        sl1.setMinimum(-100)
        sl1.setMaximum(100)
        sl1.setSingleStep(1)
        w2 = MyWidget((sl1,), name="contrast", slot=slot)

        # ROTATION
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("angle")
        sl1.setMinimum(-25)
        sl1.setMaximum(25)
        sl1.setValue(0)
        sl1.setSingleStep(1)

        combo1 = QComboBox()
        combo1.setObjectName("interpolation")
        combo1.addItem('bilinear', None)
        combo1.addItem('1 near neighbor', None)
        combo1.setCurrentIndex(0)

        w3 = MyWidget((sl1, combo1), name="rotation", slot=slot)

        # saturation
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("sat")
        sl1.setValue(0)
        sl1.setMinimum(-255)
        sl1.setMaximum(255)
        sl1.setSingleStep(5)
        w4 = MyWidget((sl1,), name="saturation", slot=slot)

        # warmth
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("warm")
        sl1.setValue(0)
        sl1.setMinimum(0)
        sl1.setMaximum(10)
        sl1.setSingleStep(1)
        w5 = MyWidget((sl1,), name="warmth", slot=slot)

        # fade
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("factor")
        sl1.setValue(0)
        sl1.setMinimum(0)
        sl1.setMaximum(10)
        sl1.setSingleStep(1)

        sl2 = QSlider(Qt.Horizontal)
        sl2.setObjectName("gray")
        sl2.setValue(0)
        sl2.setMinimum(0)
        sl2.setMaximum(255)
        sl2.setSingleStep(1)
        w6 = MyWidget((sl1, sl2), name="fade", slot=slot)

        # Highlights
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("highlight")
        sl1.setValue(0)
        sl1.setMinimum(-100)
        sl1.setMaximum(100)
        sl1.setSingleStep(1)

        w7 = MyWidget((sl1,), name="highlight", slot=slot)

        # shadow
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("shadow")
        sl1.setValue(0)
        sl1.setMinimum(-100)
        sl1.setMaximum(100)
        sl1.setSingleStep(1)

        w8 = MyWidget((sl1,), name="shadow", slot=slot)

        # Scale
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("scale")
        sl1.setMinimum(-10)
        sl1.setMaximum(-1)
        sl1.setValue(-10)
        sl1.setSingleStep(1)

        sl2 = QSlider(Qt.Horizontal)
        sl2.setObjectName("x")
        sl2.setValue(50)
        sl2.setMinimum(0)
        sl2.setMaximum(100)
        sl2.setSingleStep(1)

        sl3 = QSlider(Qt.Horizontal)
        sl3.setObjectName("y")
        sl3.setValue(50)
        sl3.setMinimum(0)
        sl3.setMaximum(100)
        sl3.setSingleStep(1)

        combo1 = QComboBox()
        combo1.setObjectName("interpolation")
        combo1.addItem('bilinear', None)
        combo1.addItem('1 near neighbor', None)
        combo1.setCurrentIndex(0)
        w9 = MyWidget((sl1, sl2, sl3, combo1), name="zoom", slot=slot)

        # Vignette
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("size")
        sl1.setMinimum(0)
        sl1.setMaximum(10)
        sl1.setValue(0)
        sl1.setSingleStep(1)

        sl2 = QSlider(Qt.Horizontal)
        sl2.setObjectName("move_h")
        sl2.setMinimum(0)
        sl2.setMaximum(10)
        sl2.setValue(5)
        sl2.setSingleStep(1)

        sl3 = QSlider(Qt.Horizontal)
        sl3.setObjectName("move_v")
        sl3.setMinimum(0)
        sl3.setMaximum(10)
        sl3.setValue(5)
        sl3.setSingleStep(1)

        cbox1 = QCheckBox()
        cbox1.setObjectName("mask_visible")
        cbox1.setChecked(False)

        w10 = MyWidget((sl1, sl2, sl3, cbox1), name="vignette", slot=slot)
        # Sharpen
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("step")
        sl1.setMinimum(0)
        sl1.setMaximum(10)
        sl1.setValue(0)
        sl1.setSingleStep(1)

        w11 = MyWidget((sl1,), name="sharpen", slot=slot)

        # Tilt shift
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("size")
        sl1.setMinimum(0)
        sl1.setMaximum(10)
        sl1.setValue(5)
        sl1.setSingleStep(1)

        sl2 = QSlider(Qt.Horizontal)
        sl2.setObjectName("move")
        sl2.setMinimum(0)
        sl2.setMaximum(10)
        sl2.setValue(5)
        sl2.setSingleStep(1)

        cbox1 = QCheckBox()
        cbox1.setObjectName("horizontal")
        cbox1.setChecked(True)

        cbox2 = QCheckBox()
        cbox2.setObjectName("mask_visible")
        cbox2.setChecked(False)

        w12 = MyWidget((sl1, sl2, cbox1, cbox2), name="tilt", slot=slot)

        tab.addTab(w, w.objectName())
        tab.addTab(w2, w2.objectName())
        tab.addTab(w3, w3.objectName())
        tab.addTab(w4, w4.objectName())
        tab.addTab(w5, w5.objectName())
        tab.addTab(w6, w6.objectName())
        tab.addTab(w7, w7.objectName())
        tab.addTab(w8, w8.objectName())
        tab.addTab(w9, w9.objectName())
        tab.addTab(w10, w10.objectName())
        tab.addTab(w11, w11.objectName())
        tab.addTab(w12, w12.objectName())
        tab.currentChanged.connect(slot)

        return tab
