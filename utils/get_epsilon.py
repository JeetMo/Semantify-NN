import numpy as np
from numba import njit, jit
import json

@njit
def get_bound(points_max, points_min, corners, colors):
    dist_max = np.maximum(1.0 - np.sqrt(np.sum(np.square(corners - points_max), axis=1)), 0.0)
    dist_min = np.maximum(1.0 - np.sqrt(np.sum(np.square(corners - points_min), axis=1)), 0.0)

    values_max = np.sum(np.multiply(colors.T, dist_max).T, axis=0)/np.sum(dist_min)
    values_min = np.sum(np.multiply(colors.T, dist_min).T, axis=0)/np.sum(dist_max)

    return values_min, values_max

# @njit
def get_inv_bound(pos, center, theta1, theta2, image):
    x = pos[0] - center[0]
    y = pos[1] - center[1]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    ts = [theta + theta1, theta + theta2]
    if np.floor(2 * ts[0] / np.pi) != np.floor(2 * ts[1] / np.pi):
        ts.append(np.pi * np.floor(2 * ts[1] / np.pi) / 2)
    ts = np.array(ts)

    x_low = r * np.amin(np.cos(ts))
    y_low = r * np.amin(np.sin(ts))
    x_high = r * np.amax(np.cos(ts))
    y_high = r * np.amax(np.sin(ts))

    xs = []
    ys = []
    if np.floor(x_low) != np.floor(x_high):
        xs.append((x_low, np.floor(x_high) - 0.00001))
        xs.append((np.floor(x_high), x_high))
    else:
        xs.append((x_low, x_high))

    if np.floor(y_low) != np.floor(y_high):
        ys.append((y_low, np.floor(y_high) - 0.00001))
        ys.append((np.floor(y_high), y_high))
    else:
        ys.append((y_low, y_high))

    b_l = np.ones_like(image[0][0])
    b_h = np.zeros_like(image[0][0])
    for x_low, x_high in xs:
        for y_low, y_high in ys:
            points = [(x_low, y_low), (x_low, y_high), (x_high, y_low), (x_high, y_high)]
            corners = []
            colors = []
            for i, p in enumerate(points):
                x1 = int(np.floor(p[0])) + int(i / 2)
                y1 = int(np.floor(p[1])) + (i % 2)
                val = image[max(min(x1 + center[0], len(image) - 1), 0)][max(min(y1 + center[1], len(image[0]) - 1), 0)]
                corners.append((x1, y1))
                colors.append(val)
            bl, bh = get_bound(np.array(points), np.array(points[::-1]), np.array(corners), np.array(colors))
            b_l = np.minimum(b_l, bl)
            b_h = np.maximum(b_h, bh)
    b_h = np.minimum(b_h, 1.0)
    b_l = np.maximum(b_l, 0.0)

    return b_h - image[pos[0]][pos[1]], b_l - image[pos[0]][pos[1]]

@njit
def get_mult_bound(colors, d_min, d_max):
    values_max = np.sum(np.multiply(colors.T, d_max).T, axis=0)/np.sum(d_min)
    values_min = np.sum(np.multiply(colors.T, d_min).T, axis=0)/np.sum(d_max)

    return values_min, values_max

def get_eps_2(image, theta1, theta2, div=1):
    rows = len(image)
    columns = len(image[0])
    center = ((rows - 1) // 2, (columns - 1) // 2)
    theta1, theta2 = np.pi * theta1 / 180, np.pi * theta2 / 180

    theta_delta = (theta2 - theta1)/div
    epsilons = []
    offsets = []
    for k in range(div):
        t1 = k*theta_delta + theta1
        t2 = (k+1)*theta_delta + theta1
        eps = np.zeros_like(image)
        offset = np.zeros_like(image)
    
        for i in range(rows):
            for j in range(columns):
                max_v, min_v = get_inv_bound((i, j), center, t1, t2, image)
                eps[i][j] += (max_v - min_v)/2
                offset[i][j] += ((max_v + min_v)/2)
        epsilons.append(eps.flatten())
        offsets.append(offset.flatten())
    return np.array(epsilons).T, np.array(offsets).T

def get_eps(image, theta1, theta2, div=1):

    rows = len(image)
    columns = len(image[0])
    center = ((rows - 1) // 2, (columns - 1) // 2)
    n = rows - 1

    epsilons = []
    offsets = []
    
    with open("rotation_multipliers/rotate_mults_theta" + str(theta1) + '.json') as infile:
        mult = json.load(infile) 
    
    for mult_spec in mult.values():    
        eps = np.zeros_like(image)
        offset = np.zeros_like(image)

        for i in range(rows):
            j_val = mult_spec[str(i - center[0])]
            for j in range(columns):
                [coords, d_min, d_max] = j_val[str(j - center[1])]
                colors = np.array([image[max(min(a + center[0],n), 0)][max(min(b + center[1],n),0)] for a,b in coords])
                d_min = np.array(d_min)
                d_max = np.array(d_max)
                
                max_v, min_v = get_mult_bound(colors, d_min, d_max)
                eps[i][j] += (max_v - min_v)/2
                offset[i][j] += ((max_v + min_v)/2) - image[i][j]
        epsilons.append(eps.flatten())
        offsets.append(offset.flatten())

    return np.array(epsilons).T, np.array(offsets).T


def get_eps_cnn(image, theta1, theta2, div=1):

    rows = len(image)
    columns = len(image[0])
    center = ((rows - 1) // 2, (columns - 1) // 2)

    n = rows - 1
    epsilons = []
    offsets = []

    with open("rotation_multipliers/rotate_mults_theta" + str(theta1) + '.json') as infile:
        mult = json.load(infile)

    for mult_spec in mult.values():
        eps = np.zeros_like(image)
        offset = np.zeros_like(image)
        for i in range(rows):
            j_val = mult_spec[str(i - center[0])]
            for j in range(columns):
                [coords, d_min, d_max] = j_val[str(j - center[1])]
                colors = np.array([image[max(min(a + center[0],n), 0)][max(min(b + center[1],n),0)] for a,b in coords])
                d_min = np.array(d_min)
                d_max = np.array(d_max)
                max_v, min_v = get_mult_bound(colors, d_min, d_max)
                eps[i][j] += (max_v - min_v)/2
                offset[i][j] += ((max_v + min_v)/2) - image[i][j]
        epsilons.append(eps.T)
        offsets.append(offset.T)

    return np.array(epsilons).T, np.array(offsets).T

def get_eps_cnn_2(image, theta1, theta2, div=1):
    rows = len(image)
    columns = len(image[0])
    center = ((rows - 1) // 2, (columns - 1) // 2)
    theta1, theta2 = np.pi * theta1 / 180, np.pi * theta2 / 180

    theta_delta = (theta2 - theta1)/div
    epsilons = []
    offsets = []
    for k in range(div):
        t1 = k*theta_delta + theta1
        t2 = (k+1)*theta_delta + theta1
        eps = np.zeros_like(image)
        offset = np.zeros_like(image)

        for i in range(rows):
            for j in range(columns):
                max_v, min_v = get_inv_bound((i, j), center, t1, t2, image)
                eps[i][j] += (max_v - min_v)/2
                offset[i][j] += ((max_v + min_v)/2)
        epsilons.append(eps.T)
        offsets.append(offset.T)
    return np.array(epsilons).T, np.array(offsets).T

import json

def get_inv_mult(x,y,theta,delta):
    theta = (np.pi*theta/180.0)
    delta = (np.pi*delta/180.0)
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x) + theta
    ts = [theta, theta + delta]
    if np.floor(2 * ts[0] / np.pi) != np.floor(2 * ts[1] / np.pi):
        ts.append(np.pi * np.floor(2 * ts[1] / np.pi) / 2)
    ts = np.array(ts)

    x_low = r * np.amin(np.cos(ts))
    y_low = r * np.amin(np.sin(ts))
    x_high = r * np.amax(np.cos(ts))
    y_high = r * np.amax(np.sin(ts))

    xs = []
    ys = []
    if np.floor(x_low) != np.floor(x_high):
        xs.append((x_low, np.floor(x_high) - 0.00001))
        xs.append((np.floor(x_high), x_high))
    else:
        xs.append((x_low, x_high))

    if np.floor(y_low) != np.floor(y_high):
        ys.append((y_low, np.floor(y_high) - 0.00001))
        ys.append((np.floor(y_high), y_high))
    else:
        ys.append((y_low, y_high))

    coord = {}
    for x_low, x_high in xs:
        for y_low, y_high in ys:
            points = [(x_low, y_low), (x_low, y_high), (x_high, y_low), (x_high, y_high)]
            corners = []
            for i, p in enumerate(points):
                x1 = int(np.floor(p[0])) + int(i / 2)
                y1 = int(np.floor(p[1])) + (i % 2)
                corners.append((x1, y1))
            points1 = np.array(points)
            points2 = np.array(points[::-1])
            corners = np.array(corners)
            div_min = np.maximum(1.0 - np.sqrt(np.sum(np.square(corners - points1), axis=1)), 0.0)
            div_max = np.maximum(1.0 - np.sqrt(np.sum(np.square(corners - points2), axis=1)), 0.0)
            for dmi, dma, corner in zip(div_min, div_max, corners):
                corner = str(tuple(corner))
                if (dmi > 0.0) and (dma > 0.0):
                    vals = coord.get(corner, [0.0, 1.0])
                    vals[0] = max(dmi, vals[0])
                    vals[1] = min(dma, vals[1])
                    coord[corner] = vals

    if len(coord) > 4:
        print(coord)
    return  coord

delta = 0.005

# @njit
def make_multipliers(x_lim, y_lim, theta):
    multipliers = {}
    for t in range(100):
        t_val = round(theta + t*delta,3)
        multipliers[t_val] = {}
        for i in range(-1*x_lim, x_lim):
            multipliers[t_val][i] = {}
            for j in range(-1*y_lim, y_lim):
                multipliers[t_val][i][j] = get_inv_mult(i, j, theta + t*delta, delta)
    with open('rotation_multipliers/rad_rotate_mults_theta' + str(theta) + '.json', 'w') as outfile:
        json.dump(multipliers, outfile)
    return multipliers


if __name__ == "__main__":
    import time

    start = time.time()
    for m in range(-360, 360):
        theta = round(0.5*m,3)
        with open('rotation_multipliers/rad_rotate_mults_theta' + str(theta) + '.json') as infile:
            mult = json.load(infile)
        new_mult = {}
        for t, k in mult.items():
            new_mult[t] = {}
            for i, ki in k.items():
                new_mult[t][i] = {}
                for j, coords in ki.items():
                    coords_l = []
                    d_min = []
                    d_max = []
                    for coord, val in coords.items():
                        coords_l.append(tuple(int(s) for s in coord.strip("()").split(",")))
                        d_min.append(val[0])
                        d_max.append(val[1])
                    new_mult[t][i][j] = [coords_l, d_min, d_max]
        with open('rotation_multipliers/rotate_mults_theta' + str(theta) + '.json', 'w') as outfile:
            json.dump(new_mult, outfile)

    print ("-------Prepocess time-------", time.time() - start)    
