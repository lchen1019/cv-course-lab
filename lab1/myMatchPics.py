import cv2 as cv
import numpy as np
import skimage.feature

from helper import plotMatches
from myHarris import harris_detector
from mySIFT import sift_descriptor
from numpy.linalg import norm


def sift_match(desc1, desc2, max_ratio=0.8):
    n = len(desc1)
    m = len(desc2)
    matches = []

    distance_matrix = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        for j in range(m):
            distance_matrix[i, j] = norm(desc1[i] - desc2[j])

    for i in range(n):
        min_inx = np.argmin(distance_matrix[i, :])
        min_value = distance_matrix[i, min_inx]

        distance_matrix[i, min_inx] = np.Inf

        second_min_inx = np.argmin(distance_matrix[i, :])
        second_min_value = distance_matrix[i, second_min_inx]

        if min_value / second_min_value < max_ratio:
            matches.append([i, min_inx])

    matches = np.array(matches)
    return matches


def matchPics(I1, I2):
    I1 = cv.cvtColor(I1, cv.COLOR_BGR2GRAY)
    I2 = cv.cvtColor(I2, cv.COLOR_BGR2GRAY)

    locs1 = harris_detector(I1)
    locs2 = harris_detector(I2)

    # Obtain descriptors for the computed feature locations
    desc1, locs1 = sift_descriptor(I1, locs1)
    desc2, locs2 = sift_descriptor(I2, locs2)

    # Match features using the descriptors
    # matches = sift_match(desc1, desc2, max_ratio=0.8)
    matches = skimage.feature.match_descriptors(desc1, desc2, 'euclidean', cross_check=True, max_ratio=0.9)

    return matches, locs1, locs2


def main():
    I1 = cv.imread('images/cv_cover.jpg')
    I2 = cv.imread('images/rotation/30.png')

    # I1 = cv.imread('images/duola1.jpg')
    # I2 = cv.imread('images/duola2.jpg')

    matches, locs1, locs2 = matchPics(I1, I2)

    plotMatches(I1, I2, matches, locs1, locs2)


if __name__ == '__main__':
    main()
