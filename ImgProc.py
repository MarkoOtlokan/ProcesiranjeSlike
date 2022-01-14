import time

import cv2
import numpy as np
import logging
from math import tan, sin, cos, pi
from scipy.interpolate import UnivariateSpline

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


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

    def func4(self, sat):
        hsv = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, sat)
        s[s > 255] = 255
        s[s < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        self.img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return True

    def func5(self, warm = 0):
        def create_LUT_8UC1(x, y, factor = 0):
            for ind in range(len(y)):
                temp = y[ind] - x[ind]
                y[ind] = x[ind] + factor * temp
            spl = UnivariateSpline(x, y)
            return spl(range(256))

        warm = warm / 10
        incr = create_LUT_8UC1([0, 50, 100, 150, 200, 245, 256], [0, 58, 124, 190, 229, 247, 256], warm)
        decr = create_LUT_8UC1([0, 50, 100, 150, 200, 245, 256], [0, 41,  80, 123, 184, 212, 246], warm)

        identity = np.arange(256, dtype=np.dtype('uint8'))
        lut = np.dstack((decr, identity, incr))
        self.img = cv2.LUT(self.orig_img, lut).astype(np.uint8)

        sat_const = warm * 50
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, sat_const)
        s[s > 255] = 255
        s[s < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        self.img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return True

    def func6(self, factor = 0, gray = 0):
        factor = 1 - (factor / 10)
        temp = self.orig_img.copy()
        temp[:, :] = (gray, gray, gray) # maybe optimize

        self.img = cv2.addWeighted(self.orig_img, factor, temp, 1 - factor, 0.0);
        return True
