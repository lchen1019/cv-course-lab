import numpy as np
import cv2 as cv
from matplotlib import gridspec, pyplot as plt
from utils import conv, gaussian_filter


def corners_visualization(size, corners):
    response_origin = np.zeros(size)
    # response_origin[response > 0] = 255

    response_l_nms = np.zeros(size)
    response_l_nms[corners[:, 0], corners[:, 1]] = 255

    gs = gridspec.GridSpec(1, 2)

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.imshow(response_origin, cmap='gray')
    ax1.imshow(response_l_nms, cmap='gray')

    ax0.axis('off')
    ax1.axis('off')

    plt.show()


def harris_detector(image):
    # gradient calculate
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = conv(image, kernel_x, mode='same')
    Iy = conv(image, kernel_y, mode='same')

    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2

    # sum
    Sxx = gaussian_filter(Ixx, sigma=1, kernel=(3, 3), mode='same')
    Sxy = gaussian_filter(Ixy, sigma=1, kernel=(3, 3), mode='same')
    Syy = gaussian_filter(Iyy, sigma=1, kernel=(3, 3), mode='same')

    k = 0.05

    height, width = Sxx.shape

    trace = Sxx + Syy
    det = Sxx * Syy - Sxy ** 2
    R = det - k * trace ** 2

    # L-NMS
    corners = []
    for h in range(5, height - 5):
        for w in range(5, width - 5):
            area = R[h - 5:h + 5, w - 5:w + 5]
            center = R[h, w]
            flag = np.where(area >= center)
            if flag[0].shape[0] == 1 and center > 0:
                corners.append([h, w])

    corners = np.array(corners)

    return corners


def main():
    location = 'images/duola1.jpg'

    image = cv.imread(location)
    image = cv.resize(image, (300, 300), interpolation=cv.INTER_CUBIC)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    corners = harris_detector(image)
    corners_visualization(image.shape, corners)


if __name__ == '__main__':
    main()
