import time

import cv2
import numpy as np
import func
from itertools import cycle
import logging
from math import tan, sin, cos, pi
from scipy.interpolate import UnivariateSpline

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
        self.blurred = None
        self.sharpen = None
        self.name2Trans = {"contrast": self.func2, "rotation": self.func3,
                           "brightness": self.func1, "saturation": self.func4,
                           "warmth": self.func5, "fade": self.func6,
                           "highlight": self.func7, "shadow": self.func8,
                           "zoom": self.func9, "sharpen": self.func11,
                           "tilt": self.func12, "vignette": self.func10
                           }

    def read_img(self, filepath):
        self.img = cv2.imread(filepath)
        self.change_orig()

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
        self.blurred = None
        self.sharpen = None
        logging.debug(f"org image changed\n")

    def func1(self, add_brightness=0):
        pixel_add = [add_brightness] * 3
        self.img = np.clip(self.orig_img + pixel_add, 0, 255).astype(np.uint8)
        return True

    def func2(self, p=0):
        #self.func4(round(p * (-1) * 2.55))
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
        self.img = contrast[self.orig_img]#self.img]
        #lut = np.dstack((contrast, contrast, contrast))
        #self.img = apply_lut(self.img, lut)
        return True

    def func3(self, angle=0, scale=1.0):
        scale /= 10
        center_point = np.array(self.orig_img.shape[:2]) / 2
        logging.debug(f'specific point: {center_point}')
        self.img = func.rotate(self.orig_img, angle, center_point, scale)
        return True

    def saturation_helper(self, org_img, sat):
        hsv = func.rgb_to_hsv_vectorized(org_img)
        hsv[..., 1] = np.clip(hsv[..., 1] + [sat], 0, 255).astype(np.uint8)
        self.img = func.hsv_to_bgr_vectorized(hsv)

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
        #self.saturation_helper(self.img, sat_const) # satu too slow and warmth good without it
        return True

    def func6(self, factor=0, gray=0):
        factor = 1 - (factor / 10)
        self.img = func.add_weighted(self.orig_img, gray, factor)
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
        shape = self.orig_img.shape[:2]
        y_max, x_max = shape
        specific_point = np.rint([y*y_max/100, x*x_max/100])
        logging.debug(f'shape point: {y_max, x_max}')
        logging.debug(f'specific point: {specific_point}')
        self.img = func.rotate(self.orig_img, 0, specific_point, scale)
        return True

    def func10(self, move_h, move_v, size):
        size /= 10
        move_h /= 10
        move_v /= 10
        mask = func.radial_mask(self.orig_img.shape[:2], (move_h, move_v), size)
        self.img = np.rint(self.orig_img * mask).astype(np.uint8)
        return True

    def func11(self, step=10, to_sharpen=True):
        self.img = self.orig_img
        if to_sharpen:
            step /= 10
            if self.sharpen is None:
                self.sharpen = func.apply_kernel(self.orig_img, func.sharpen)
            self.img = func.add_weighted(self.sharpen, self.orig_img, step)
        return True

    def func12(self, size=10, move=0, horizontal=True):
        size /= 10
        move /= 10
        if self.blurred is None:
            self.blurred = func.apply_kernel(self.orig_img, func.blur2)
        #self.img = func.add_weighted(blurred_img, self.orig_img, linear)
        mask = func.linear_mask(self.orig_img.shape[:2], move, size, horizontal)
        self.img = np.rint(self.orig_img * mask + self.blurred * (1 - mask)).astype(np.uint8)#np.rint(self.orig_img * mask).astype(np.uint8)
        return True

