from PyQt5.QtWidgets import (QVBoxLayout, QWidget,
                             QSlider, QCheckBox, QPushButton, QLabel, QComboBox, QHBoxLayout)


class MyWidget(QWidget):
    def __init__(self, el, name="name", slot=None):
        super().__init__()
        self.setObjectName(name)
        self.slot = slot
        self.lay = QVBoxLayout()
        self.setLayout(self.lay)
        self.init_el(el, self.lay)

    def init_el(self, el, layout):
        for e in el:
            if isinstance(e, QSlider):
                e.valueChanged.connect(self.slot)
            elif isinstance(e, QCheckBox):
                e.stateChanged.connect(self.slot)
            elif isinstance(e, QPushButton):
                e.clicked.connect(self.slot)
            elif isinstance(e, QComboBox):
                e.activated.connect(self.slot)
            elif isinstance(e, QLabel):
                layout.addWidget(e)
                continue
            else:
                raise Exception("MyWidget class got unexpected element!!")
            e.setParent(self)
            layout.addWidget(e)

    def give_slot(self):
        self.slot(**self.give_vals())

    def give_vals(self):
        d = {}
        for e in self.children():
            if isinstance(e, QSlider):
                d[e.objectName()] = e.value()
            elif isinstance(e, QCheckBox):
                d[e.objectName()] = e.isChecked()
            elif isinstance(e, QComboBox):
                d[e.objectName()] = e.itemText(e.currentIndex())
        return d



