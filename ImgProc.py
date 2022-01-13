import time

import cv2
import numpy as np
import logging
from math import tan, sin, cos, pi

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


class PP:
    def __init__(self):
        self.img = None
        self.orig_img = None
        self.name2Trans = {"contrast": self.func2,
                           "brightness": self.func1}

    def read_img(self, filepath):
        self.img = cv2.imread(filepath)
        self.orig_img = self.img

    # Ako transform vraca None ako se funkcija ne moze odraditi
    # u suprotnom vraca modifikovanu sliku.
    # Funkcije koje transform zove vracaju True i False/None
    # ako su uspesno/neuspesno izvrsile
    def transform(self, t, **pars):
        if self.name2Trans[t](**pars):
            return self.img

    def trans(self):
        return self.name2Trans.keys()

    def change_orig(self):
        self.orig_img = self.img
        logging.debug(f"org image changed\n")

    def func1(self, add_brightness=0):
        fun = cv2.add
        if add_brightness < 0:
            fun = cv2.subtract
            add_brightness = abs(add_brightness)
        addition = np.zeros_like(self.orig_img)
        pixel_add = [add_brightness] * 3
        addition[:, :] = pixel_add
        self.img = fun(self.orig_img, addition)
        return True

    def func2(self, p=0):
        p = p / 100

        def help(x):
            res = None
            if x < 127.5 * cos(pi * (1 - p) / 4):
                res = tan(pi * (1 - p) / 4) * x
            elif 127.5 * cos(pi * (1 - p) / 4) <= x <= 255 - 127.5 * cos(pi * (1 - p) / 4):
                res = ((1 - sin(pi * (1 - p) / 4)) / (1 - cos(pi * (1 - p) / 4))) * (
                        x - 127.5 * cos(pi * (1 - p) / 4)) + 127.5 * sin(pi * (1 - p) / 4)
            elif x > 255 - 127.5 * cos(pi * (1 - p) / 4):
                res = tan(pi * (1 - p) / 4) * (x - 255 + 127.5 * cos(pi * (1 - p) / 4)) + 255 - 127.5 * sin(
                    pi * (1 - p) / 4)
            return int(res)
        abc = [help(x) for x in range(256)]
        logging.debug(f"lut table: {abc}\n")
        contrast = np.array(abc, dtype=np.uint8)
        # contrast = np.array([ (i-74)*p+74 for i in range (0,256)]).clip(0,255).astype('uint8')
        # mada ovo nije moj kod...
        # p (float): How much to adjust the contrast. Can be any
        #             non negative number. 0 gives a solid gray image, 1 gives the
        #             original image while 2 increases the contrast by a factor of 2.
        lut = contrast #np.dstack((contrast, contrast, contrast))
        self.img = cv2.LUT(self.orig_img, lut)
        return True

    def func3(self):
        None
