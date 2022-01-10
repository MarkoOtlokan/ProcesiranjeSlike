import time

import cv2
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

class PP:
    def __init__(self):
        self.img = None
        self.orig_img = None
        self.name2Trans = {"saturation": self.func1,
                           "brightness": self.func2}
        self.last_t = None

        self.bright = 0
        self.sat = 0

    def read_img(self, filepath):
        self.img = cv2.imread(filepath)
        self.orig_img = self.img

    # Ako transform vraca None ako se funkcija ne moze odraditi
    # u suprotnom vraca modifikovanu sliku.
    # Funkcije koje transform zove vracaju True i False/None
    # ako su uspesno/neuspesno izvrsile
    def transform(self, t, **pars):
        if not self.last_t == t:  # ako je nova transf original sliku promeni
            self.orig_img = self.img

        if self.name2Trans[t](**pars):
            self.last_t = t
            return self.img

    def trans(self):
        return self.name2Trans.keys()

    def func1(self, sat=0):
        sat = sat - self.sat

        hsv = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, sat)
        s[s > 255] = 255
        s[s < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        new_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        self.img = new_img
        return True

    def func2(self, add_brightness=0):
        add_brightness = add_brightness - self.bright #temporary hack


        hsv = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, add_brightness)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        new_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        self.img = new_img
        return True

    def func3(self):
        None
