import logging

import numpy as np

idk = np.array([[0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]])

sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

sharpen2 = np.array([[-1, -1, -1],
                     [-1, 9, -1],
                     [-1, -1, -1]])

blur = np.array([[0.0625, 0.125, 0.0625],
                 [0.125, 0.25, 0.125],
                 [0.0625, 0.125, 0.0625]])

blur2 = (1 / 9) * np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])


def add_weighted(img, img2, factor):
    return ((img * factor) + (img2 * (1 - factor))).astype('uint8')


def radial_mask(shape, move, size):  # return in range 0-1
    # size in range (0, 1)
    h, w = shape
    size = (1.2 - 0.5) * size + 0.5
    move_h, move_v = move
    move_h = 2 * move_h - 1
    move_v = 2 * move_v - 1
    Y = np.linspace(-1 + move_h, 1 + move_h, w)[None, :]
    X = np.linspace(-1 + move_v, 1 + move_v, h)[:, None]
    alpha = np.power(X ** 2 + Y ** 2, size)
    alpha = 1 - alpha / alpha.max()
    return alpha[..., None] * np.array([1, 1, 1])


def lin_grad(n, center=0.5, size_white=0.1, transition=0.1):
    arr = np.zeros(n)
    center_i = round(center * n)
    len_white = round(size_white * n)
    start_white_i = center_i - len_white // 2
    end_white_i = center_i + len_white // 2

    len_t = round(transition * n / 2)
    trans = np.linspace(0, 1, len_t)
    start_t_i = start_white_i - len_t
    end_t_i = end_white_i + len_t

    indexes = np.array([start_t_i, start_white_i, end_white_i, end_t_i])
    # indexes = np.clip(indexes, 0, n - 1)

    logging.debug(f'indexes : {indexes}')
    arr[indexes[1]:indexes[2]] = 1
    arr[indexes[0]:indexes[1]] = trans
    arr[indexes[2]:indexes[3]] = trans[::-1]

    return arr


def linear_mask(shape, move, size, horizontal=True):  # return in range 0-1
    # size in range (0, 1)
    # move in range (0, 1)
    h, w = shape
    size = (0.2 - 0.05) * size + 0.05
    move = (0.85 - 0.15) * move + 0.15
    if horizontal:
        h, w = w, h
    X = lin_grad(w, move, size)
    if horizontal:
        X = np.tile(X[..., None], (1, h))
    else:
        X = np.tile(X, (h, 1))
    return X[..., None] * np.array([1, 1, 1])


def rgb_to_hsv_vectorized(img):  # input img with BGR format
    maxc = img.max(-1)
    minc = img.min(-1)

    out = np.zeros(img.shape)
    out[:, :, 2] = maxc
    out[:, :, 1] = ((maxc - minc) / maxc) * 255

    divs = (maxc[..., None] - img) / ((maxc - minc)[..., None])
    cond1 = divs[..., 0] - divs[..., 1]
    cond2 = 2.0 + divs[..., 2] - divs[..., 0]
    h = 4.0 + divs[..., 1] - divs[..., 2]
    mask2 = img[..., 2] == maxc
    mask1 = img[..., 1] == maxc
    h[mask2] = cond1[mask2]
    h[mask1] = cond2[mask1]
    out[:, :, 0] = ((h / 6.0) % 1.0) * 180

    out[minc == maxc, :2] = 0
    return np.rint(out).astype(np.uint8)


def hsv_to_bgr_vectorized(img):  # return img with BGR format
    c = ((img[..., 2] / 255) * (img[..., 1] / 255))
    h_sec = img[..., 0] / 30
    x = c * (1 - np.abs(np.fmod(h_sec, 2) - 1))
    m = (img[..., 2] / 255) - c
    z = np.zeros(c.shape)

    r = np.empty(c.shape)
    g = np.empty(c.shape)
    b = np.empty(c.shape)

    mask = [[c, x, z],
            [x, c, z],
            [z, c, x],
            [z, x, c],
            [x, z, c],
            [c, z, x]]

    hi = np.mod(np.floor(h_sec), 6)

    for i, (X, Y, Z) in enumerate(mask):
        indices = hi == i
        b[indices] = X[indices]
        g[indices] = Y[indices]
        r[indices] = Z[indices]

    out = (np.dstack((r, g, b)) + m[..., None]) * 255
    return np.rint(out).astype(np.uint8)


def getRotationMatrix2D(center, angle, scale):
    cX, cY = center
    angle = angle * np.pi / 180
    warp_mat = np.zeros((2, 3))
    a = np.cos(angle) * scale
    b = np.sin(angle) * scale
    warp_mat[:2, :2] = [[a, b], [-b, a]]
    warp_mat[:2, 2] = center - np.matmul(warp_mat[:2, :2], center)
    return warp_mat


def warpAffine(I, M, interpolation='bilinear'):
    x, y = I.shape[:2]
    Xi, Yi = np.mgrid[0:x, 0:y]
    Xi = Xi.ravel().astype(np.uint16)
    Yi = Yi.ravel().astype(np.uint16)

    Inew_index_x, Inew_index_y = M[0, 0] * Xi + M[0, 1] * Yi + M[0, 2], M[1, 0] * Xi + M[1, 1] * Yi + M[1, 2]

    img = np.zeros(I.shape)

    l1and2mask = (0 <= Inew_index_x) & (Inew_index_x <= x-1) & (Inew_index_y <= y-1) & (0 <= Inew_index_y)

    Xi = Xi[l1and2mask]
    Yi = Yi[l1and2mask]
    Inew_index_x = Inew_index_x[l1and2mask]
    Inew_index_y = Inew_index_y[l1and2mask]
    New_pixels = None
    if interpolation == "bilinear":
        New_pixels = bilinear_interpolation(Inew_index_x, Inew_index_y, I)
    else:
        New_pixels = I[Inew_index_x.astype(np.uint16), Inew_index_y.astype(np.uint16)]
    img[Xi, Yi] = New_pixels
    return img.astype(np.uint8)


def bilinear_interpolation(idx1, idx2, img):
    x0 = (1 - idx1 % 1)[:, None] * [1, 1, 1]
    x1 = (idx1 % 1)[:, None] * [1, 1, 1]

    y0 = (1 - idx2 % 1)[:, None] * [1, 1, 1]
    y1 = (idx2 % 1)[:, None] * [1, 1, 1]

    floor_x = np.floor(idx1).astype(np.uint16)
    floor_y = np.floor(idx2).astype(np.uint16)

    ceil_x = np.ceil(idx1).astype(np.uint16)
    ceil_y = np.ceil(idx2).astype(np.uint16)

    q11 = img[floor_x, floor_y]
    q12 = img[floor_x, ceil_y]
    q21 = img[ceil_x, floor_y]
    q22 = img[ceil_x, ceil_y]

    return x0 * y0 * q11 + x1 * y0 * q21 + x0 * y1 * q12 + x1 * y1 * q22


def apply_kernel(img, kernel):
    k = len(kernel)
    p = k // 2
    padded = np.pad(img.copy(), ((p, p), (p, p), (0, 0)), "symmetric").astype(np.float32)  # zbog negativnih mora signed
    img_x, img_y = padded.shape[:2]

    padded[p:-p, p:-p, :] = np.add.reduce([padded[x:(img_x - (k - x - 1)), y:(img_y - (k - y - 1)), :] * kernel[x, y]
                                           for x in range(k) for y in range(k)])
    return np.clip(padded[p:-p, p:-p, :], 0, 255).astype(np.uint8)