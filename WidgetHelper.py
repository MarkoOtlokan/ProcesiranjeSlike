from PyQt5.QtWidgets import (QVBoxLayout, QWidget,
                             QSlider, QCheckBox, QPushButton)


class MyWidget(QWidget):
    def __init__(self, *el, name="name", slot=None):
        super().__init__()
        self.setObjectName(name)
        self.slot = slot
        self.el = el
        self.lay = QVBoxLayout()
        self.setLayout(self.lay)
        self.init_el()

    def init_el(self):
        for e in self.el:
            if isinstance(e, QSlider):
                self.init_slider(e)
            elif isinstance(e, QCheckBox):
                self.init_checkbox(e)
            elif isinstance(e, QPushButton):
                self.init_button(e)
            else:
                raise Exception("MyWidget class got unexpected element!!")
            e.setParent(self)

    def init_slider(self, e):
        self.lay.addWidget(e)
        e.valueChanged.connect(lambda x: self.slot(**self.give_vals()))

    def init_checkbox(self, e):
        self.lay.addWidget(e)
        e.stateChanged.connect(lambda: self.slot(**self.give_vals()))

    def init_button(self, e):
        self.lay.addWidget(e)
        e.clicked.connect(lambda: self.slot(**self.give_vals()))

    def reset_gadgets(self):
        for e in self.el:
            if isinstance(e, QSlider):
                e.setValue(0)

    def give_vals(self):
        d = {}
        for e in self.el:
            if isinstance(e, QSlider):
                d[e.objectName()] = e.value()
            elif isinstance(e, QCheckBox):
                d[e.objectName()] = e.isChecked()
        return d

