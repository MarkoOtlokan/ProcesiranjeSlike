import logging

import numpy as np

idk = np.array([[0 , -1,  0],
                [-1,  4, -1],
                [0 , -1,  0]])

sharpen = np.array([[0 , -1,  0],
                    [-1,  5, -1],
                    [0 , -1,  0]])

blur = np.array([[0.0625, 0.125, 0.0625],
                 [0.125 , 0.25 , 0.125 ],
                 [0.0625, 0.125, 0.0625]])

bilinear = np.array([[0   , 0.25,    0],
                     [0.25, 1   , 0.25],
                     [0   , 0.25, 0   ]])

bilinear2 = np.array([[0.25  , 0.5, 0.25],
                     [0.5    , 1  , 0.5 ],
                     [0.25   , 0.5, 0.25]])

# sharpen3d = np.dstack((sharpen, sharpen, sharpen))

def add_weighted(img, img2, factor):
    return ((img * factor) + (img2 * (1 - factor))).astype('uint8')


def radial_mask(shape, move, size):  # return in range 0-1
    # size in range (0, 1)
    h, w = shape
    size = (1.2 - 0.5) * size + 0.5
    move_h, move_v = move
    move_h = 2 * move_h - 1
    move_v = 2 * move_v - 1
    Y = np.linspace(-1+move_h, 1+move_h, w)[None, :]
    X = np.linspace(-1+move_v, 1+move_v, h)[:, None]
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
    X = np.tile(X, (h, 1))
    #alpha = 1 - alpha # / alpha.max()
    if horizontal:
        X = X.T
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


def rotate(img, angle, point, scale=1.0):
    angle = angle * np.pi / 180
    warp_mat = np.zeros((2, 3))
    a, b = np.cos(angle) * scale, np.sin(angle) * scale
    warp_mat[:2, :2] = [[a, b], [-b, a]]
    logging.debug(f'warp_mat: {warp_mat}')
    logging.debug(f'point: {point}')
    logging.debug(f'mat mult: {np.matmul(point, warp_mat[:2, :2])}')
    logging.debug(f'mat mult2: {np.matmul(warp_mat[:2, :2], point)}')
    warp_mat[:2, 2] = point - np.matmul(warp_mat[:2, :2], point)
    return warpAffine(img, warp_mat)


def warpAffine(I, M):
    a, b = M[..., :2].T, M[..., 2]
    x, y = I.shape[:2]
    Xi, Yi = np.mgrid[0:x, 0:y]
    Xi = Xi.ravel().astype(np.uint16)
    Yi = Yi.ravel().astype(np.uint16)
    Iindex = np.column_stack((Xi, Yi))
    Inew_index = np.matmul(Iindex, a) + b
    img = np.zeros(I.shape)

    l1 = Inew_index[..., 0].astype(np.uint16)
    l2 = Inew_index[..., 1].astype(np.uint16)
    l1and2mask = (l1 < x) & (l2 < y)
    l1 = l1[l1and2mask]
    l2 = l2[l1and2mask]
    Xi = Xi[l1and2mask]
    Yi = Yi[l1and2mask]
    # logging.debug(f'I shape type: {I.shape}')
    # logging.debug(f'img shape type: {img.shape}')
    # logging.debug(f'l1 shape type: {l1.shape}')
    # logging.debug(f'Xi shape type: {Xi.shape}')

    img[Xi, Yi] = I[l1, l2]
    return img.astype(np.uint8)


def apply_kernel(img, kernel):
    k = len(kernel)
    p = k // 2
    padded = np.pad(img.copy(), ((p, p), (p, p), (0, 0)), "symmetric").astype(np.int16)  # zbog negativnih mora signed
    img_x, img_y = padded.shape[:2]

    padded[p:-p, p:-p, :] = np.add.reduce([padded[x:(img_x - (k - x - 1)), y:(img_y - (k - y - 1)), :] * kernel[x, y]
                                           for x in range(k) for y in range(k)])
    return np.clip(padded[p:-p, p:-p, :], 0, 255).astype(np.uint8)
