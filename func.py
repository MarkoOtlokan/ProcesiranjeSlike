import numpy as np


def rgb_to_hsv_vectorized(img): # input img with BGR format
    maxc = img.max(-1)
    minc = img.min(-1)

    out = np.zeros(img.shape)
    out[:,:,2] = maxc
    out[:,:,1] = ((maxc-minc) / maxc) * 255

    divs = (maxc[...,None] - img)/ ((maxc-minc)[...,None])
    cond1 = divs[...,0] - divs[...,1]
    cond2 = 2.0 + divs[...,2] - divs[...,0]
    h = 4.0 + divs[...,1] - divs[...,2]
    h[img[...,2]==maxc] = cond1[img[...,2]==maxc]
    h[img[...,1]==maxc] = cond2[img[...,1]==maxc]
    out[:,:,0] = ((h/6.0) % 1.0) * 180

    out[minc == maxc,:2] = 0
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