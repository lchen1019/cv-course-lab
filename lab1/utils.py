import numpy as np
from math import sqrt, pi, exp


def gaussian_func(sigma, x, y):
    """return Gaussian(sigma, sqrt(x^2 + y^2))"""
    d = x ** 2 + y ** 2
    left = 1 / (sigma * sqrt(2 * pi))
    right = exp(-0.5 * d / (sigma ** 2))
    return left * right


def conv(x, kernel, mode):
    """convolution"""
    kernel_x, kernel_y = kernel.shape
    res = None
    if mode == 'same':
        res = np.zeros(x.shape)
        left = int(kernel_x / 2)
        right = kernel_x - left - 1
        top = int(kernel_y / 2)
        down = kernel_y - top - 1
        x = np.pad(x, ((top, down), (left, right)))
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                temp = x[i:i+kernel_x, j:j+kernel_y]
                res[i, j] = np.sum(temp * kernel)
    elif mode == 'valid':
        res = np.zeros((x.shape[0] - kernel_x + 1, x.shape[1] - kernel_y[1] + 1))
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                temp = x[i:i+kernel_x, j:j+kernel_y]
                x[i, j] = np.sum(temp * kernel)
    return res


def gaussian_filter(x, sigma=1, kernel=(3, 3), mode="same"):
    """convolution using gaussian filter"""
    left = 1 / (sigma * sqrt(2 * pi))
    gaussian_kernel = np.zeros(kernel)
    for i in range(kernel[0]):
        for j in range(kernel[1]):
            d = i ** 2 + j ** 2
            right = exp(-0.5 * d / (sigma ** 2))
            gaussian_kernel[i, j] = left * right
    gaussian_kernel = gaussian_kernel * 1 / np.sum(gaussian_kernel)
    return conv(x, gaussian_kernel, mode)


if __name__ == '__main__':
    sigma = 1.5
    for i in range(8):
        for j in range(8):
            print(gaussian_func(sigma, i, j))
