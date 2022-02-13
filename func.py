import logging

import numpy as np

sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

blur = np.array([[0.0625, 0.125, 0.0625],
                 [0.125 , 0.25 , 0.125 ],
                 [0.0625, 0.125, 0.0625]])


# sharpen3d = np.dstack((sharpen, sharpen, sharpen))

def add_weighted(img, img2, factor):
    return ((img * factor) + (img2 * (1 - factor))).astype('uint8')

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
    h[img[..., 2] == maxc] = cond1[img[..., 2] == maxc]
    h[img[..., 1] == maxc] = cond2[img[..., 1] == maxc]
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
    warp_mat[:2, :2] = [[a, -b], [b, a]]
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
    padded = np.pad(img.copy(), ((p, p), (p, p), (0, 0)), "symmetric").astype(np.int16) # zbog negativnih mora signed
    img_x, img_y = padded.shape[:2]

    padded[p:-p, p:-p, :] = np.add.reduce([padded[x:(img_x - (k - x - 1)), y:(img_y - (k - y - 1)), :] * kernel[x, y]
                                           for x in range(k) for y in range(k)])
    return np.clip(padded[p:-p, p:-p, :], 0, 255).astype(np.uint8)
