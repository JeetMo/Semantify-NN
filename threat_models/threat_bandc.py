import numpy as np


def get_first_layers(image, w1=None, b1=None):

    i = image.flatten()
    w_1 = np.vstack((i, np.ones_like(i)))
    w_1 = np.hstack((w_1, w_1)).T
    b_1 = np.hstack((0.5 * np.ones_like(i), -0.5 * np.ones_like(i))).T
    w_2 = np.hstack((np.eye(i.shape[0]), -1.0 * np.eye(i.shape[0])))
    b_2 = -0.5*np.ones_like(i)

    ws = [np.copy(w_1, 'C').astype(np.float32), np.copy(np.matmul(w1, w_2), 'C').astype(np.float32)]
    bs = [np.copy(b_1, 'C').astype(np.float32), np.copy(np.dot(w1, b_2) + b1, 'C').astype(np.float32)]

    return ws, bs


def get_first_layers_cnn(image):
    rows = image.shape[0]
    columns = image.shape[1]
    if len(image.shape) < 3:
        channels = 1
    else:
        channels = image.shape[2]

    w_1 = np.ones((rows, columns, 2*channels, rows, columns, 2)).astype(np.float32)
    b_1 = np.zeros((rows, columns, 2*channels))
    p_1 = (rows - 1, rows - 1, columns - 1, columns - 1)
    s_1 = (1, 1)
    w_2 = np.zeros((rows, columns, channels, 1, 1, 2*channels))
    b_2 = -0.5*np.ones((rows, columns, channels))
    p_2 = (0, 0, 0, 0)
    s_2 = (1, 1)
    shapes = [(1, 1, 1), (rows, columns, 5)]

    for r in range(rows):
        for c in range(columns):
            for j in range(channels):
                w_2[r, c, j, 0, 0, 2*j: 2*(j+1)] += np.array([1.0, -1.0])

            b_1[r, c, :] = np.array([0.0, - 1.0, 0.0, - 1.0, 0.0, - 1.0])

    ws = [w_1.astype(np.float32), w_2.astype(np.float32)]
    bs = [b_1.astype(np.float32), b_2.astype(np.float32)]

    return ws, bs, shapes, [p_1, p_2], [s_1, s_2]