import numpy as np
import cv2 as cv
import scipy.optimize

from loader import *
from numpy import float32, float64


def _singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))

    return F


def _objective_F(f, pts1, pts2):
    F = _singularize(f.reshape([3, 3]))
    num_points = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
    hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
    Fp1 = F.dot(hpts1.T)
    FTp2 = F.T.dot(hpts2.T)

    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1)) ** 2 * (1 / (fp1[0] ** 2 + fp1[1] ** 2) + 1 / (fp2[0] ** 2 + fp2[1] ** 2))

    return r


def refineF(F, pts1, pts2):
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
        maxiter=100000,
        maxfun=10000
    )

    return _singularize(f.reshape([3, 3]))


def calc_normalized_matrix(h, w):
    """calculate the normalized matrix to make the image in range (0, 1)"""
    matrix = np.array([
        [1 / w, 0, 0],
        [0, 1 / h, 0],
        [0, 0, 1]
    ], dtype=float32)
    return matrix


def normalized(normalized_matrix, points):
    result = np.zeros_like(points, dtype=float32)
    for inx, point in enumerate(points):
        point_homogenous = np.array([point[0], point[1], 1], dtype=float32)
        result[inx] = normalized_matrix.dot(point_homogenous)[:2]
    return result


def calc_M9_matrix(pts1, pts2):
    """return the M * 9 matrix"""
    n = pts1.shape[0]
    M = np.zeros((n, 9), dtype=float32)
    for i in range(n):
        M[i, 0] = pts2[i, 0] * pts1[i, 0]
        M[i, 1] = pts2[i, 0] * pts1[i, 1]
        M[i, 2] = pts2[i, 0]

        M[i, 3] = pts2[i, 1] * pts1[i, 0]
        M[i, 4] = pts2[i, 1] * pts1[i, 1]
        M[i, 5] = pts2[i, 1]

        M[i, 6] = pts1[i, 0]
        M[i, 7] = pts1[i, 1]
        M[i, 8] = 1
    return M


def eight_points_algorithm(pts1, pts2, M1, M2):
    """using SVD to solve the fundamental matrix"""

    # normalize all points in range (-1, 1) by the following matrix
    normalized_matrix1 = calc_normalized_matrix(M1[0], M1[1])
    normalized_matrix2 = calc_normalized_matrix(M2[0], M2[1])

    pts1 = normalized(normalized_matrix1, pts1)
    pts2 = normalized(normalized_matrix2, pts2)

    # get the M * 9 matrix
    M = calc_M9_matrix(pts1, pts2)

    # SVD: the corresponding eigenvector of the minimum eigenvalues M^T * M
    A = np.dot(M.T, M)
    eig_value, eig_vector = np.linalg.eig(A)
    inx = np.argmin(eig_value)
    F = eig_vector[:, inx].reshape(3, 3)

    # rank 2 constraint
    u, sigma, v = np.linalg.svd(F)
    s = np.zeros((3, 3), dtype=float32)
    sigma[-1] = 0
    row, col = np.diag_indices_from(s)
    s[row, col] = sigma
    F = u.dot(s).dot(v)

    # refine F
    F = refineF(F, pts1, pts2)

    # un-normalize fundamental matrix
    F = normalized_matrix2.T.dot(F).dot(normalized_matrix1)

    return F


def similarity(v1, v2, mode='SSD'):
    v1 = v1.reshape(-1, 1)
    v2 = v2.reshape(-1, 1)
    if mode == 'SAD':
        return np.sum(np.abs(v1 - v2))
    elif mode == 'SSD':
        return np.sum((v1 - v2) ** 2)
    elif mode == 'NCC':
        v1 = v1 - np.average(v1)
        v2 = v2 - np.average(v2)
        up = np.sum(v1 * v2)
        down = np.sqrt(np.sum(v1 ** 2) * np.sum(v2 ** 2))
        return 1.0 - float64(up) / float64(down)


def epipolar_correspondences(img1, img2, F, pts1):
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    d = 3
    y_len, x_len = img1.shape
    padding = 2 * d
    img1 = np.pad(img1, ((padding, padding), (padding, padding)), 'constant', constant_values=((0, 0), (0, 0)))
    img2 = np.pad(img2, ((padding, padding), (padding, padding)), 'constant', constant_values=((0, 0), (0, 0)))

    epsilon = 10 ** -100
    pts2 = np.zeros_like(pts1)

    for inx, point in enumerate(pts1):
        point = np.array([point[0], point[1], 1])
        L = F.dot(point)
        a, b, c = L

        patch1 = img1[point[1]+padding-d:point[1]+padding+d+1, point[0]+padding-d:point[0]+padding+d+1]

        distance = 10 ** 10
        for x in range(0, x_len):
            y = int(-(c + a * x) / (b + epsilon))
            if y < 0 or y > y_len:
                continue
            patch2 = img2[y + padding - d:y + padding + d + 1, x + padding - d:x + padding + d + 1]
            # print(patch1.shape, patch2.shape)
            temp = similarity(patch1, patch2, 'NCC')
            if temp < distance:
                distance = temp
                pts2[inx][:] = [x, y]

    return pts2


def essential_matrix(F, K1, K2):
    return K2.T.dot(F).dot(K1)


def triangulate(P1, pts1, P2, pts2):
    n = pts1.shape[0]
    M = np.empty((4, 4), dtype=float64)
    pts3d = np.zeros((n, 3), dtype=float64)
    pts1 = pts1.astype(float64)
    pts2 = pts2.astype(float64)

    for i in range(n):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        M[0, :] = y1 * P1[2, :] - P1[1, :]
        M[1, :] = P1[0, :] - x1 * P1[2, :]
        M[2, :] = y2 * P2[2, :] - P2[1, :]
        M[3, :] = P2[0, :] - x2 * P2[2, :]

        A = np.dot(M.T, M)
        eig_value, eig_vector = np.linalg.eig(A)
        inx = np.argmin(eig_value)
        pts3d[i] = eig_vector[:, inx][:3] / eig_vector[:, inx][3]

    return pts3d


