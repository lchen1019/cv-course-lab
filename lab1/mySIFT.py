import math
import numpy as np
import cv2 as cv

from math import atan2
from myHarris import harris_detector
from lab1.utils import gaussian_func, gaussian_filter


def gradient(x, y, image):
    dx = float(image[x, y + 1]) - float(image[x, y - 1])
    dy = float(image[x + 1, y]) - float(image[x - 1, y])
    return atan2(dy, dx) * (180 / np.pi)


def to_bins(piece):
    bins = [0] * 8
    for i in piece:
        for j in i:
            bins[int(j / 45)] += 1
    return bins


def get_main_gradient(sigma, x, y, image_gradient):
    R = (3 * sigma * math.sqrt(2) * 5 + 1) / 2
    gaussian_sigma = 1.5 * sigma

    # 构造求解主方向直方图
    bins = [0] * 36
    for i in range(x - int(R), x + int(R)):
        for j in range(y - int(R), y + int(R)):
            if (x - i) ** 2 + (y - j) ** 2 <= R ** 2:
                point_weight = gaussian_func(gaussian_sigma, (x - i), (y - j))
                bins[int(image_gradient[i][j] / 10)] += point_weight
    bins = np.array(bins)
    bins = bins / np.sum(bins)
    # 求最大与次最大
    max_inx = 0
    next_max_inx = -1
    for i in range(1, len(bins)):
        if bins[i] > bins[max_inx]:
            next_max_inx = max_inx
            max_inx = i
        elif bins[next_max_inx] >= bins[i]:
            next_max_inx = i
    # print(bins)
    # print(max_inx, next_max_inx)

    # 超过80%，构造两个
    if bins[next_max_inx] < 0.8 * bins[max_inx]:
        return [max_inx * 10]
    return [max_inx * 10, next_max_inx * 10]


def sift_descriptor(image, corners):
    image = gaussian_filter(image, sigma=1, kernel=(3, 3), mode='same')
    # 可调参
    sigma = 1

    # 求解每一个像素点的方向
    height, width = image.shape
    image_gradient = np.empty(image.shape)
    padding = 20
    for h in range(1, height - 1):
        for w in range(1, width - 1):
            image_gradient[h, w] = ((gradient(h, w, image) % 360) + 360) % 360
    image_gradient = np.pad(image_gradient, ((padding, padding), (padding, padding)))

    # 遍历每一个角点，得到其16 * 16邻域
    descriptors = []
    corners_after = []
    for h, w in corners:
        h = h + padding
        w = w + padding
        patch = image_gradient[h-8: h+8, w-8: w+8]

        # 归一化到主轴方向
        for main_direction in get_main_gradient(sigma, h, w, image_gradient):
            patch = (((patch - main_direction) % 360) + 360) % 360

            # 划分到8个bins
            descriptor = []
            for m in range(4):
                for n in range(4):
                    piece = patch[4*m: 4*(m+1), 4*n: 4*(n+1)]
                    bins = to_bins(piece)
                    descriptor += bins
            descriptor = np.array(descriptor)
            descriptor = descriptor / sum(descriptor)
            descriptors.append(descriptor)
            corners_after.append([h - padding, w - padding])

    descriptors = np.array(descriptors)
    corners_after = np.array(corners_after)

    return descriptors, corners_after


def main():
    location = 'images/duola1.jpg'

    image = cv.imread(location)
    image = cv.resize(image, (300, 300), interpolation=cv.INTER_CUBIC)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    corners = harris_detector(image)
    descriptors, corners = sift_descriptor(image, corners)


if __name__ == '__main__':
    main()
