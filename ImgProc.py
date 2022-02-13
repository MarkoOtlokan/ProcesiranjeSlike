import time

import cv2
import numpy as np
import func
from itertools import cycle
import logging
from math import tan, sin, cos, pi
from scipy.interpolate import UnivariateSpline
import math

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


def create_lut(x, y, factor=0, my_range=(0, 256)):
    for ind in range(len(y)):
        temp = y[ind] - x[ind]
        y[ind] = x[ind] + factor * temp
    spl = UnivariateSpline(x, y)
    return spl(range(*my_range)).astype(np.uint8)


def apply_lut(img, lut):
    img2 = img.copy()
    for i in range(img2.shape[2]):
        img2[:, :, i] = lut[0, :, i][img2[:, :, i]]
    return img2


class PP:
    def __init__(self):
        self.img = None
        self.orig_img = None
        self.name2Trans = {"contrast": self.func2, "rotation": self.func3,
                           "brightness": self.func1, "saturation": self.func4,
                           "warmth": self.func5, "fade": self.func6,
                           "highlight": self.func7, "shadow": self.func8,
                           "zoom": self.func9, "vignatte": self.func10,
                           }

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
        pixel_add = [add_brightness] * 3
        self.img = np.clip(self.orig_img + pixel_add, 0, 255).astype(np.uint8)
        return True

    def func2(self, p=0):
        self.func4(round(p * (-1) * 2.55))
        p = p / 100
        p += 1

        def help(y):
            if p > 1:
                abcd = None
                if y < 128:
                    abcd = (y - 128 * p) * p + 128
                else:
                    abcd = (y - 128) * p + p * 128
                return abcd
            return (y - 128) * p + 128

        contrast = np.array([help(i) for i in range(0, 256)]).clip(0, 255).astype('uint8')
        logging.debug(f"lut table: {contrast}\n")
        #self.img = np.vectorize(dict(enumerate(contrast)).get)(self.img)
        #self.img = np.take(contrast, self.img)
        self.img = contrast[self.img]
        #lut = np.dstack((contrast, contrast, contrast))
        #self.img = apply_lut(self.img, lut)
        return True

    def func3(self, angle=0, scale=1.0):
        scale /= 10
        center_point = np.array(self.orig_img.shape[:2][::-1]) / 2
        logging.debug(f'specific point: {center_point}')
        self.img = func.rotate(self.orig_img, angle, center_point, scale)
        return True

    def saturation_helper(self, org_img, sat):
        def non_overflowing_sum(a, b):
            c = np.int16(a) + b
            c[np.where(c > 255)] = 255
            c[np.where(c < 0)] = 0
            return np.uint8(c)

        hsv = func.rgb_to_hsv_vectorized(org_img)
        h, s, v = np.transpose(hsv, (2, 0, 1))
        s = non_overflowing_sum(s, sat)

        final_hsv = np.dstack((h, s, v))
        self.img = func.hsv_to_bgr_vectorized(final_hsv)

    def func4(self, sat):
        self.saturation_helper(self.orig_img, sat)
        return True

    def func5(self, warm=0):
        warm = warm / 10
        incr = create_lut([0, 50, 100, 150, 200, 245, 256], [0, 58, 124, 190, 229, 247, 256], warm)
        decr = create_lut([0, 50, 100, 150, 200, 245, 256], [0, 41, 80, 123, 184, 212, 246], warm)
        identity = np.arange(256, dtype=np.dtype('uint8'))

        lut = np.dstack((decr, identity, incr))
        self.img = apply_lut(self.orig_img, lut)

        sat_const = warm * 50
        self.saturation_helper(self.img, sat_const)
        return True

    def func6(self, factor=0, gray=0):
        factor = 1 - (factor / 10)
        fade = np.zeros_like(self.orig_img)
        fade[:, :] = (gray, gray, gray)
        self.img = ((self.orig_img * factor) + (fade * (1 - factor))).astype('uint8')
        return True

    def func7(self, highlight=0, pixel=128):
        highlight = highlight / 100
        logging.debug(f' pixel value: {pixel}')
        fromm = [128, 138, 150, 180, 210, 240, 253, 255]
        too = [128, 149, 174, 219, 238, 248, 252, 255]
        if highlight < 0:
            too = [128, 135, 141, 153, 179, 201, 230, 243]

        non_change = np.arange(pixel).astype(np.uint8)
        changed = create_lut(fromm, too, abs(highlight), (pixel, 256))
        highlightValue = np.concatenate((non_change, changed))
        logging.debug(f"lut table: {highlightValue}\n")
        self.img = highlightValue[self.orig_img]
        return True

    def func8(self, shadow=0, pixel=128):
        shadow = shadow / 100
        logging.debug(f' pixel value: {pixel}')
        fromm = [0, 20, 40, 60, 80, 100, 120, 125, 128]
        too = [0, 16, 21, 30, 37, 49, 75, 111, 128]
        if shadow < 0:
            too = [0, 22, 50, 80, 109, 119, 126, 127, 128]

        non_change = np.arange(pixel, 256).astype(np.uint8)
        changed = create_lut(fromm, too, abs(shadow), (0, pixel))
        shadowValue = np.concatenate((changed, non_change))
        logging.debug(f"lut table: {shadowValue}\n")
        self.img = shadowValue[self.orig_img]
        return True

    def func9(self, scale=1.0, x=0, y=0):
        scale /= -10
        shape = self.orig_img.shape[:2][::-1]
        y_max, x_max = shape
        specific_point = np.rint([y*y_max/100, x*x_max/100])
        logging.debug(f'specific point: {specific_point}')
        self.img = func.rotate(self.orig_img, 0, specific_point, scale)
        return True

    def func10(self, vig=0):#RADI ALI JE SPORO
        newImage = self.orig_img.copy()
        h = len(newImage)
        w = len(newImage[0])
        h2 = h/2
        w2 = w/2

        for i in range (0, h):
            for j in range (0, w):
                piksel = []
                piksel = newImage[i][j]
                #vrednost = ((abs(h2 - i) + abs(w2 - j)) * vig) / 100
                vrednost = (math.sqrt(pow((h2-i),2) + pow((w2 - j),2)) * vig)/100
                for x in range(0,3):
                    piksel[x] = max(piksel[x] - vrednost, 0)
                #newImage[i][j] = piksel

        self.img = newImage
        return True





















        return True
