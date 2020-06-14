import numpy as np


def get_first_layers(image, w1=None, b1=None):
    factors = np.zeros_like(image)

    rows = image.shape[0]
    columns = image.shape[1]
    if len(image.shape) < 3:
        channels = 1
    else:
        channels = image.shape[2]

    lightness_p = np.zeros((2 * rows * columns,))
    lightness_n = np.zeros((2 * rows * columns,))

    weights = []
    biases = []
    for r in range(rows):
        for c in range(columns):
            colors = image[r][c]
            l = (max(colors) + min(colors)) / 2 + 0.5
            f = 1 - abs(1 - 2 * l)
            if f == 0:
                cx = np.zeros_like(colors)
            else:
                cx = np.array([(c + 0.5 - l) / f for c in colors])

            factors[r][c] = cx

            lightness_p[2 * (r * columns + c)] = 1 - l
            lightness_p[2 * (r * columns + c) + 1] = 1 - 2 * l

            lightness_n[2 * (r * columns + c)] = l
            lightness_n[2 * (r * columns + c) + 1] = 2 * l - 1

    factors = factors.flatten()

    pw_1 = -1.0 * np.ones((1, 2 * rows * columns))
    for i in range(rows * columns):
        pw_1[0][2 * i + 1] = -2.0
    pbias_1 = lightness_p
    pw_2 = np.zeros((2 * rows * columns, channels * rows * columns))
    pb_2 = np.ones((channels * rows * columns,)) - 0.5
    for i in range(rows * columns):
        for ind in range(channels):
            pw_2[2 * i][channels * i + ind] = 2.0 * factors[channels * i + ind] - 1.0
            pw_2[2 * i + 1][channels * i + ind] = -2.0 * factors[channels * i + ind]

    nw_1 = np.ones((1, 2 * rows * columns))
    for i in range(rows * columns):
        nw_1[0][2 * i + 1] = 2.0
    nbias_1 = lightness_n
    nw_2 = np.zeros((2 * rows * columns, channels * rows * columns))
    nb_2 = np.zeros((channels * rows * columns,)) - 0.5
    for i in range(rows * columns):
        for ind in range(channels):
            nw_2[2 * i][channels * i + ind] = 1.0 + 2.0 * factors[channels * i + ind]
            nw_2[2 * i + 1][channels * i + ind] = -2.0 * factors[channels * i + ind]

    #ws = {"pos": [pw_1.T.astype(np.float32), pw_2.T.astype(np.float32)],
    #      "neg": [nw_1.T.astype(np.float32), nw_2.T.astype(np.float32)]}
    #bs = {"pos": [pbias_1.astype(np.float32), pb_2.astype(np.float32)],
    #      "neg": [nbias_1.astype(np.float32), nb_2.astype(np.float32)]}
    #
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

    w_1 = []
    for r in range(rows):
        w_1.append([])
        for c in range(columns):
            val = [np.ones((rows, columns, 1)), np.ones((rows, columns, 1)), 2 * np.ones((rows, columns, 1)), 2 * np.ones((rows, columns, 1)),
                   2 * np.ones((rows, columns, 1))]
            w_1[r].append(val)
    w_1 = np.array(w_1).astype(np.float32)
    b_1 = np.zeros((rows, columns, 5))
    p_1 = (rows - 1, rows - 1, columns - 1, columns - 1)
    s_1 = (1, 1)
    w_2 = np.zeros((rows, columns, channels, 1, 1, 5))
    b_2 = -0.5*np.ones((rows, columns, channels))
    p_2 = (0, 0, 0, 0)
    s_2 = (1, 1)
    shapes = [(1, 1, 1), (rows, columns, 5)]

    for r in range(rows):
        for c in range(columns):
            colors = image[r][c]
            l = (max(colors) + min(colors)) / 2 + 0.5
            f = 1 - abs(1 - 2 * l)
            if f == 0:
                cx = np.zeros_like(colors)
            else:
                cx = np.array([(c + 0.5 - l) / f for c in colors])

            for j in range(channels):
                w_2[r, c, j, 0, 0, :] += np.array([1.0, -1.0, cx[j], -2*cx[j], cx[j]])

            b_1[r, c, :] = np.array([l, l - 1.0, 2*l, 2*l - 1.0, 2*l - 2.0])

    ws = [w_1.astype(np.float32), w_2.astype(np.float32)]
    bs = [b_1.astype(np.float32), b_2.astype(np.float32)]

    return ws, bs, shapes, [p_1, p_2], [s_1, s_2]
