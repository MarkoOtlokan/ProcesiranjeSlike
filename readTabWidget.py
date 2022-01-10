from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, \
    QSlider, QPushButton
from PyQt5.QtCore import Qt

from WidgetHelper import MyWidget

# test....
class ReadTabW:
    def getTab(slot, change_im):
        tab = QTabWidget()

        #for ... read in all widgets from file
        #just example
        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("add_brightness")
        sl1.setValue(0)
        sl1.setMinimum(-400)
        sl1.setMaximum(400)
        sl1.setSingleStep(10)

        w = MyWidget(sl1, name="brightness", slot=slot)### VAZNO!!! MyWidget name mora odgovarati
                                                       ### nazivu funkcije u mapping-u ImgProc.PP klasi.
                                                       ### Takodje gadgeti unutar (ovog gore)
                                                       ### moraju imati imena = parametrima
                                                       ### funkcije cije ime deli (taj unutar kojeg se nalaze) Mywidget!

        sl1 = QSlider(Qt.Horizontal)
        sl1.setObjectName("sat")
        sl1.setValue(0)
        sl1.setMinimum(-400)
        sl1.setMaximum(400)
        sl1.setSingleStep(10)
        w2 = MyWidget(sl1, name="saturation", slot=slot)

        tab.addTab(w, w.objectName())
        tab.addTab(w2, w2.objectName())

        def temp_slot(_):
            change_im()
            tab.currentWidget().reset_gadgets()

        tab.currentChanged.connect(temp_slot)

        return tab
