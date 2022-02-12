from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, \
    QSlider, QPushButton
from PyQt5.QtCore import Qt

from WidgetHelper import MyWidget

# test....
class ReadTabW:
    def getTab(slot, change_pixelmap, change_orig):
        tab = QTabWidget()

        #for ... read in all widgets from file
        #just example
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("add_brightness")
        sl1.setValue(0)
        sl1.setMinimum(-255)
        sl1.setMaximum(255)
        sl1.setSingleStep(10)

        w = MyWidget(sl1, name="brightness", slot=slot)### VAZNO!!! MyWidget name mora odgovarati
                                                       ### nazivu funkcije u mapping-u ImgProc.PP klasi.
                                                       ### Takodje gadgeti unutar (ovog gore)
                                                       ### moraju imati imena = parametrima
                                                       ### funkcije cije ime deli (taj unutar kojeg se nalaze) Mywidget!

        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("p")
        sl1.setValue(0)
        sl1.setMinimum(-100)
        sl1.setMaximum(100)
        sl1.setSingleStep(1)
        w2 = MyWidget(sl1, name="contrast", slot=slot)

        # ROTATION
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("angle")
        sl1.setValue(0)
        sl1.setMinimum(-25)
        sl1.setMaximum(25)
        sl1.setSingleStep(1)

        sl2 = QSlider(Qt.Horizontal)
        sl2.setObjectName("scale")
        sl2.setValue(10)
        sl2.setMinimum(5)
        sl2.setMaximum(20)
        sl2.setSingleStep(1)
        w3 = MyWidget(sl1, sl2, name="rotation", slot=slot)

        # saturation
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("sat")
        sl1.setValue(0)
        sl1.setMinimum(-255)
        sl1.setMaximum(255)
        sl1.setSingleStep(5)
        w4 = MyWidget(sl1, name="saturation", slot=slot)

        # warmth
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("warm")
        sl1.setValue(0)
        sl1.setMinimum(0)
        sl1.setMaximum(10)
        sl1.setSingleStep(1)
        w5 = MyWidget(sl1, name="warmth", slot=slot)

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
        w6 = MyWidget(sl1, sl2, name="fade", slot=slot)

        # Highlights
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("highlight")
        sl1.setValue(0)
        sl1.setMinimum(-100)
        sl1.setMaximum(100)
        sl1.setSingleStep(1)
        w7 = MyWidget(sl1, name="highlight", slot=slot)

        # Shadows
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("shadow")
        sl1.setValue(0)
        sl1.setMinimum(-100)
        sl1.setMaximum(100)
        sl1.setSingleStep(1)
        w8 = MyWidget(sl1, name="shadow", slot=slot)

        tab.addTab(w, w.objectName())
        tab.addTab(w2, w2.objectName())
        tab.addTab(w3, w3.objectName())
        tab.addTab(w4, w4.objectName())
        tab.addTab(w5, w5.objectName())
        tab.addTab(w6, w6.objectName())
        tab.addTab(w7, w7.objectName())
        tab.addTab(w8, w8.objectName())

        def temp_slot(i):
            change_pixelmap()
            tab.widget(i).give_slot()

        tab.currentChanged.connect(temp_slot)

        return tab
