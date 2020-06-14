import numpy as np


def get_first_layers(image, w1=None, b1=None):
    factors = np.zeros_like(image)
    lightness = np.zeros_like(image)

    rows = image.shape[0]
    columns = image.shape[1]
    if len(image.shape) < 3:
        channels = 1
    else:
        channels = image.shape[2]

    saturate_p = np.zeros((rows * columns,))
    saturate_n = np.zeros((rows * columns,))

    for r in range(rows):
        for c in range(columns):
            colors = image[r][c]
            l = (max(colors) + min(colors)) / 2.0 + 0.5
            s = (max(colors) - min(colors))

            if s == 0:
                cx = np.zeros_like(colors)
            else:
                f = 1.0 - abs(1.0 - 2.0 * l)
                s = s / f
                cx = np.array([(c + 0.5 - l) / s for c in colors])

            factors[r][c] = cx
            saturate_p[r * columns + c] = 1.0 - s
            saturate_n[r * columns + c] = s
            lightness[r][c] = np.ones_like(colors) * (l - 0.5)

    factors = factors.flatten()
    lightness = lightness.flatten()

    pw_1 = -1.0 * np.ones((1, rows * columns))
    pbias_1 = saturate_p
    pw_2 = np.zeros((rows * columns, channels * rows * columns))
    pb_2 = np.array(lightness)
    for i in range(rows * columns):
        for ind in range(channels):
            pw_2[i][channels * i + ind] = -1 * factors[channels * i + ind]
            pb_2[channels * i + ind] += factors[channels * i + ind]

    nw_1 = np.ones((1, rows * columns))
    nbias_1 = saturate_n
    nw_2 = np.zeros((rows * columns, channels * rows * columns))
    nb_2 = lightness
    for i in range(rows * columns):
        for ind in range(channels):
            nw_2[i][channels * i + ind] = factors[channels * i + ind]

    ws = {"pos": [pw_1.T.astype(np.float32), np.matmul(w1, pw_2.T).astype(np.float32)],
          "neg": [nw_1.T.astype(np.float32), np.matmul(w1, nw_2.T).astype(np.float32)]}
    bs = {"pos": [pbias_1.astype(np.float32), (b1 + np.dot(w1, pb_2)).astype(np.float32)],
          "neg": [nbias_1.astype(np.float32), (b1 + np.dot(w1, nb_2)).astype(np.float32)]}

    return ws, bs


def get_first_layers_cnn(image):
    rows = image.shape[0]
    columns = image.shape[1]
    if len(image.shape) < 3:
        channels = 1
    else:
        channels = image.shape[2]

    w_1 = np.ones((rows, columns, 2, rows, columns, 1))
    b_1 = np.zeros((rows, columns, 2))
    p_1 = (rows - 1, rows - 1, columns - 1, columns - 1)
    s_1 = (1, 1)
    w_2 = np.zeros((rows, columns, channels, 1, 1, 2))
    b_2 = -0.5*np.ones((rows, columns, channels))
    p_2 = (0, 0, 0, 0)
    s_2 = (1, 1)
    shapes = [(1, 1, 1), (rows, columns, 2)]

    for r in range(rows):
        for c in range(columns):
            colors = image[r][c]
            l = (max(colors) + min(colors)) / 2.0 + 0.5
            s = (max(colors) - min(colors))

            if s == 0:
                cx = np.zeros_like(colors)
            else:
                f = 1.0 - abs(1.0 - 2.0 * l)
                s = s / f
                cx = np.array([(c + 0.5 - l) / s for c in colors])

            for j in range(channels):
                w_2[r, c, j, 0, 0, :] += np.array([cx[j], -1.0*cx[j]])

            b_1[r, c, :] = np.array([s, s - 1.0])
            b_2[r, c, :] = np.array([l - 0.5, l - 0.5, l - 0.5])

    ws = [w_1.astype(np.float32), w_2.astype(np.float32)]
    bs = [b_1.astype(np.float32), b_2.astype(np.float32)]

    return ws, bs, shapes, [p_1, p_2], [s_1, s_2]