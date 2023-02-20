import cv2 as cv

from helper import computeBrief, briefMatch, plotMatches
from helper import fast_corner_detection as corner_detection


def matchPics(I1, I2):
    # I1, I2 : Images to match
    # Convert Images to GrayScale
    I1 = cv.cvtColor(I1, cv.COLOR_BGR2GRAY)
    I2 = cv.cvtColor(I2, cv.COLOR_BGR2GRAY)

    # Detect Features in Both Images
    locs1 = corner_detection(I1)
    locs2 = corner_detection(I2)

    # Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1, locs1)
    desc2, locs2 = computeBrief(I2, locs2)

    # Match features using the descriptors
    matches = briefMatch(desc1, desc2)

    return matches, locs1, locs2


def main():
    I1 = cv.imread('images/rotation/0.png')
    I2 = cv.imread('images/rotation/30.png')

    matches, locs1, locs2 = matchPics(I1, I2)
    plotMatches(I1, I2, matches, locs1, locs2)


if __name__ == '__main__':
    main()
