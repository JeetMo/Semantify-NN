import numpy as np

indices = {0: ((1,5,7,11),(2,4,8,10)), 1:((1,3,7,9), (0,4,6,10)), 2:((3,5,9,11),(2,6,8,12))}
off = [0,1,1]


def get_first_layers(image, w1=None, b1=None):
    rows = image.shape[0]
    columns = image.shape[1]
    if len(image.shape) < 3:
        channels = 1
    else:
        channels = image.shape[2]

    w_1 = np.ones((13 * rows * columns, 1))
    b_1 = np.zeros((13 * rows * columns,))
    w_2 = np.zeros((channels * rows * columns, 13 * rows * columns))
    b_2 = np.zeros((channels * rows * columns,))

    for r in range(rows):
        for c in range(columns):
            colors = image[r][c]

            delta = np.max(colors) - np.min(colors)
            if delta == 0:
                h = 0
            else:
                cmax = np.argmax(colors)
                if cmax == 0:
                    h = ((colors[1] - colors[2]) / delta) % 6
                elif cmax == 1:
                    h = ((colors[2] - colors[0]) / delta + 2)
                else:
                    h = ((colors[0] - colors[1]) / delta + 4)

            m = np.min(colors)

            for i in range(13):
                b_1[13 * (r * columns + c) + i] = h - (i - 3)

            for j in range(3):
                pos, neg = indices[j]
                for p in pos:
                    w_2[3 * (r * columns + c) + j][13 * (r * columns + c) + p] = delta
                for n in neg:
                    w_2[3 * (r * columns + c) + j][13 * (r * columns + c) + n] = -1 * delta
                b_2[3 * (r * columns + c) + j] = m + off[j] * delta


    ws = [w_1.astype(np.float32), np.matmul(w1, w_2).astype(np.float32)]
    bs = [b_1.astype(np.float32), (b1 + np.dot(w1, b_2)).astype(np.float32)]

    return ws, bs


def get_first_layers_cnn(image):
    rows = image.shape[0]
    columns = image.shape[1]
    if len(image.shape) < 3:
        channels = 1
    else:
        channels = image.shape[2]

    w_1 = np.ones((rows, columns, 13, rows, columns, 1))
    b_1 = np.zeros((rows, columns, 13))
    p_1 = (rows-1, rows-1, columns-1, columns-1)
    s_1 = (1, 1)
    w_2 = np.zeros((rows, columns, channels, 1, 1, 13))
    b_2 = np.zeros((rows, columns, channels))
    p_2 = (0,0,0,0)
    s_2 = (1, 1)

    for r in range(rows):
        for c in range(columns):
            colors = image[r][c]

            delta = np.max(colors) - np.min(colors)
            if delta == 0:
                h = 0
            else:
                cmax = np.argmax(colors)
                if cmax == 0:
                    h = ((colors[1] - colors[2]) / delta) % 6
                elif cmax == 1:
                    h = ((colors[2] - colors[0]) / delta + 2)
                else:
                    h = ((colors[0] - colors[1]) / delta + 4)

            m = np.min(colors)

            for i in range(13):
                b_1[r, c, i] = h - (i - 3)

            for j in range(3):
                pos, neg = indices[j]
                for p in pos:
                    w_2[r, c, j, 0, 0, p] = delta
                for n in neg:
                    w_2[r, c, j, 0, 0, n] = -1 * delta
                b_2[r, c, j] = m + off[j] * delta

    ws = [w_1.astype(np.float32), w_2.astype(np.float32)]
    bs = [b_1.astype(np.float32), b_2.astype(np.float32)]
    shapes = [(1, 1, 1), (rows, columns, 13)]

    return ws, bs, shapes, [p_1, p_2], [s_1, s_2]