import time

import cv2
import numpy as np
import func
from itertools import cycle
import logging
from math import tan, sin, cos, pi
from scipy.interpolate import UnivariateSpline

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


def create_lut(x, y, factor=0):
    for ind in range(len(y)):
        temp = y[ind] - x[ind]
        y[ind] = x[ind] + factor * temp
    spl = UnivariateSpline(x, y)
    return spl(range(256)).astype(np.uint8)


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
                           "warmth": self.func5, "fade": self.func6}

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

    def func3(self, angle=0, scale=0):
        # treba ovako nesto manuelno odraditi....
        def rotate_bound(image, angle):
            # grab the dimensions of the image and then determine the
            # center
            (h, w) = image.shape[:2]
            (cX, cY) = (w / 2, h / 2)

            # grab the rotation matrix (applying the negative of the
            # angle to rotate clockwise), then grab the sine and cosine
            # (i.e., the rotation components of the matrix)
            M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY

            # perform the actual rotation and return the image
            return cv2.warpAffine(image, M, (nW, nH))

        def rotate(image, angle, center=None, scale=1.0):
            # grab the dimensions of the image
            (h, w) = image.shape[:2]

            # if the center is None, initialize it as the center of
            # the image
            if center is None:
                center = (w // 2, h // 2)

            # perform the rotation
            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(image, M, (w, h))

            # return the rotated image
            return rotated

        scale = (scale / 10) + 1
        self.img = rotate(self.orig_img, angle, scale=scale)
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
