import cv2
import numpy as np
import func
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


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

    # brightness
    def func1(self, add_brightness):
        # add_brightness [-255, 255]
        pixel_add = [add_brightness] * 3
        self.img = np.clip(self.orig_img + pixel_add, 0, 255).astype(np.uint8)
        return True

    # contrast
    def func2(self, p):
        # p [-100, 100]
        p = p / 100
        p += 1

        def help(y):
            if p > 1:
                abcd = None
                if y < 128:
                    abcd = (y - 128 * p) * p + 128 # tamniji -> jos tamniji
                else:
                    abcd = (y - 128) * p + p * 128 # svetliji -> jos svetliji
                return abcd
            return (y - 128) * p + 128 # svi ka sivoj

        contrast = np.array([help(i) for i in range(0, 256)]).clip(0, 255).astype('uint8')
        logging.debug(f"lut table: \n{contrast}\n")
        self.img = contrast[self.orig_img]
        return True

    # rotacija
    def func3(self, angle, interpolation):
        # angle [-25, 25]
        # interpolation = #{'bilinear', '1 near neighbor'}
        center_point = np.array(self.orig_img.shape[:2]) / 2
        logging.debug(f'specific point: {center_point}')
        warp_mat = func.getRotationMatrix2D(center_point, angle, 1.0)
        self.img = func.warpAffine(self.orig_img, warp_mat, interpolation)
        return True

    # saturacija
    def saturation_helper(self, org_img, sat):
        hsv = func.rgb_to_hsv_vectorized(org_img)
        hsv[..., 1] = np.clip(hsv[..., 1] + [sat], 0, 255).astype(np.uint8)
        self.img = func.hsv_to_bgr_vectorized(hsv)

    # saturacija
    def func4(self, sat):
        # sat [-255, 255]
        self.saturation_helper(self.orig_img, sat)
        return True

    # warmth
    def func5(self, warm):
        # warm [0, 10]
        warm = warm / 10
        incr = func.create_lut([0, 50, 100, 150, 200, 245, 256], [0, 58, 124, 190, 229, 247, 256], warm)
        decr = func.create_lut([0, 50, 100, 150, 200, 245, 256], [0, 41, 80, 123, 184, 212, 246], warm)
        identity = np.arange(256, dtype=np.dtype('uint8'))

        lut = np.dstack((decr, identity, incr))  # BGR!
        self.img = func.apply_lut(self.orig_img, lut)
        return True

    # fade
    def func6(self, factor, gray):
        # factor [0, 10]
        # gray [0, 255]
        factor = 1 - (factor / 10)
        self.img = func.add_weighted(self.orig_img, gray, factor)
        return True

    # highlight
    def func7(self, highlight):
        # highlight [-100, 100]
        highlight = highlight / 100
        fromm = [128, 138, 150, 180, 210, 240, 253, 255]
        too = [128, 149, 174, 219, 238, 248, 252, 255]
        if highlight < 0:
            too = [128, 135, 141, 153, 179, 201, 230, 243]

        non_change = np.arange(128).astype(np.uint8)
        changed = func.create_lut(fromm, too, abs(highlight), (128, 256))
        highlightValue = np.concatenate((non_change, changed))
        logging.debug(f"lut table: \n{highlightValue}\n")
        self.img = highlightValue[self.orig_img]
        return True

    # shadow
    def func8(self, shadow):
        # shasdow [-100, 100]
        shadow = shadow / 100
        fromm = [0, 20, 40, 60, 80, 100, 120, 125, 128]
        too = [0, 16, 21, 30, 37, 49, 75, 111, 128]
        if shadow < 0:
            too = [0, 22, 50, 80, 109, 119, 126, 127, 128]

        non_change = np.arange(128, 256).astype(np.uint8)
        changed = func.create_lut(fromm, too, abs(shadow), (0, 128))
        shadowValue = np.concatenate((changed, non_change))
        logging.debug(f"lut table: \n{shadowValue}\n")
        self.img = shadowValue[self.orig_img]
        return True

    # zoom
    def func9(self, scale, x, y, interpolation):
        # scale [-10, -1]
        # x, y [0, 100]
        # interpolation = #{'bilinear', '1 near neighbor'}
        scale /= -10
        shape = self.orig_img.shape[:2]
        y_max, x_max = shape
        center_point = np.rint([y * y_max / 100, x * x_max / 100])
        warp_mat = func.getRotationMatrix2D(center_point, 0, scale)
        self.img = func.warpAffine(self.orig_img, warp_mat, interpolation)
        return True

    # vignette
    def func10(self, move_h, move_v, size, mask_visible):
        # move_h, move_v, size [0, 10]
        size /= 10
        move_h /= 10
        move_v /= 10
        mask = func.radial_mask(self.orig_img.shape[:2], (move_h, move_v), size) # mask [0, 1]
        logging.debug(f'mask values \n: {mask}')
        if not mask_visible:
            self.img = np.rint(self.orig_img * mask).astype(np.uint8)
        else:
            self.img = np.rint(255 * mask).astype(np.uint8)
        return True

    # sharpen
    def func11(self, step):
        # step [0, 10]
        self.img = self.orig_img
        step /= 10
        if self.sharpen is None:
            self.sharpen = func.apply_kernel(self.orig_img, func.sharpen3)
        self.img = func.add_weighted(self.sharpen, self.orig_img, step)
        return True

    # tilt
    def func12(self, size, move, horizontal, mask_visible):
        # size, move [0, 10]
        # horizontal #{True, False}
        size /= 10
        move /= 10
        if self.blurred is None:
            self.blurred = func.apply_kernel(self.orig_img, func.blur3)
        mask = func.linear_mask(self.orig_img.shape[:2], move, size, horizontal)
        if not mask_visible:
            self.img = np.rint(self.orig_img * mask + self.blurred * (1 - mask)).astype(np.uint8)
        else:
            self.img = np.rint(255 * mask).astype(np.uint8)
        return True
