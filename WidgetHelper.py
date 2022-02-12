from PyQt5.QtWidgets import (QVBoxLayout, QWidget,
                             QSlider, QCheckBox, QPushButton)


class MyWidget(QWidget):
    def __init__(self, *el, name="name", slot=None):
        super().__init__()
        self.setObjectName(name)
        self.slot = slot
        self.lay = QVBoxLayout()
        self.setLayout(self.lay)
        self.init_el(el)

    def init_el(self, el):
        for e in el:
            if isinstance(e, QSlider):
                e.valueChanged.connect(self.give_slot)
            elif isinstance(e, QCheckBox):
                e.stateChanged.connect(self.give_slot)
            elif isinstance(e, QPushButton):
                e.clicked.connect(self.give_slot)
            else:
                raise Exception("MyWidget class got unexpected element!!")
            e.setParent(self)
            self.lay.addWidget(e)

    def give_slot(self):
        self.slot(**self.give_vals())

    def give_vals(self):
        d = {}
        for e in self.children():
            if isinstance(e, QSlider):
                d[e.objectName()] = e.value()
            elif isinstance(e, QCheckBox):
                d[e.objectName()] = e.isChecked()
        return d

    # def reset_gadgets(self):
    #     for e in self.children():
    #         if isinstance(e, QSlider):
    #             e.setValue(0)



