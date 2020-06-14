import numpy as np


polar = {(r/10): [] for r in range(370)}

for i in range(-25, 25):
    for j in range(-25, 25):
        phi = np.arctan2(j, i)
        r = round(np.sqrt(i**2 + j**2), 1)
        polar[r].append((phi, (i, j)))


def rotate_img(image, angle):

    angle = np.pi*angle/180
    rows = len(image)
    columns = len(image[0])
    center = ((rows - 1) // 2, (columns - 1) // 2)

    def get_pos(x, y, ang):
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        new_theta = theta + ang

        x_1 = center[0] + r * np.cos(new_theta)
        y_1 = center[1] + r * np.sin(new_theta)

        x_low, x_high = int(np.floor(x_1)), int(np.ceil(x_1)) + 1
        y_low, y_high = int(np.floor(y_1)), int(np.ceil(y_1)) + 1

        value = np.zeros_like(image[0][0])
        scale = 0
        for i in range(x_low, x_high):
            for j in range(y_low, y_high):
                value += max(0, (1 - np.sqrt((i - x_1)**2 + (j - y_1)**2))) * image[max(min(i, rows-1),0)][max(min(j, rows-1),0)]
                scale += max(0, (1 - np.sqrt((i - x_1)**2 + (j - y_1)**2)))

        if scale == 0:
            return value
        return (1/scale)*value

    new_image = np.zeros_like(image)

    for r in range(rows):
        for c in range(columns):
            value = get_pos(r-center[0], c-center[1], angle)
            new_image[r][c] = value

    return new_image


def lighten_img(image, lighten):
    rows = len(image)
    columns = len(image[0])

    new_image = np.zeros_like(image)
    for r in range(rows):
        for c in range(columns):
            colors = image[r][c]
            l = 0.5 + ((max(colors) + min(colors))/2)
            l_new = max(min(1, l + lighten), 0)
            f = 1 - abs(2*l - 1)
            f_new = 1 - abs(2*l_new - 1)
            if f == 0:
                new_image[r][c] = np.array([l_new + color  - l  for color in colors])
            else:
                new_image[r][c] = np.array([l_new + (color + 0.5 - l)*f_new/f - 0.5 for color in colors])
    return new_image


def saturate_img(image, saturate):
    rows = len(image)
    columns = len(image[0])

    new_image = np.zeros_like(image)
    for r in range(rows):
        for c in range(columns):
            colors = image[r][c]
            l = 0.5 + ((max(colors) + min(colors)) / 2)
            f = 1 - abs(2 * l - 1)
            if f == 0:
                s = 0
            else:
                s = (max(colors) - min(colors))/f

            s_new = max( min(s + saturate, 1), 0)
            if s == 0:
                new_image[r][c] = np.array(colors)
            else:
                new_image[r][c] = np.array(
                    [(l + (color + 0.5 - l) * s_new / s) - 0.5 for color in colors])
    return new_image


def hue_img(image, hue):
    rows = len(image)
    columns = len(image[0])

    new_image = np.zeros_like(image)
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

            h_new = (h + hue) % 6
            x = 1 - abs((h_new % 2) - 1.0)

            if h_new < 1.0:
                new_image[r][c] = np.array([delta + m, delta*x + m, m])
            elif h_new < 2.0:
                new_image[r][c] = np.array([delta*x + m, delta+ m, m])
            elif h_new < 3.0:
                new_image[r][c] = np.array([m, delta + m, delta*x + m])
            elif h_new < 4.0:
                new_image[r][c] = np.array([m, delta*x + m, delta + m])
            elif h_new < 5.0:
                new_image[r][c] = np.array([delta*x + m, m, delta + m])
            else:
                new_image[r][c] = np.array([delta + m, m, delta * x + m])

    return new_image


def bandc_img(image, brightness, contrast):
    rows = len(image)
    columns = len(image[0])

    new_image = np.zeros_like(image)
    for r in range(rows):
        for c in range(columns):
            colors = image[r][c]

            new_colors = (1 + contrast)*colors + brightness + 0.5
            new_colors = np.minimum(np.maximum(new_colors, 0.0), 1.0) - 0.5

            new_image[r][c] = new_colors
    return new_image


def grid_attack(image, range_min, range_max, delta, method="rotate"):
    if method == "rotate":
        l, u = int(range_min / delta), 1 + int(range_max / delta)
        images = [image] + [rotate_img(image, t*delta) for t in range(l, u)]
    elif method == "lighten":
        l, u = int(range_min / delta), 1 + int(range_max / delta)
        images = [image] + [lighten_img(image, t*delta) for t in range(l, u)]
    elif method == "saturate":
        l, u = int(range_min / delta), 1 + int(range_max / delta)
        images = [image] + [saturate_img(image, t*delta) for t in range(l, u)]
    elif method == "hue":
        l, u = int(range_min / delta), 1 + int(range_max / delta)
        images = [image] + [hue_img(image, t*delta) for t in range(l, u)]
    elif method == "bandc":
        delta_1, delta_2 = delta
        l1, u1 = int(range_min / delta_1), 1 + int(range_max / delta_1)
        l2, u2 = int(range_min / delta_2), 1 + int(range_max / delta_2)
        images = [image] + [bandc_img(image, t1*delta_1, t2*delta_2) for t1 in range(l1, u1) for t2 in range(l2, u2)]
    else:
        raise ValueError
    return np.array(images)