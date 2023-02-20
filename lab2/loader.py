import numpy as np


def load_correspond():
    data = np.load('data/some_corresp.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']
    n = pts1.shape[0] if pts1.shape[0] < pts2.shape[0] else pts2.shape[0]
    return pts1[:n], pts2[:n], n


def load_template():
    data = np.load('data/temple_coords.npz')
    return data['pts1']


def load_intrinsics():
    data = np.load('data/intrinsics.npz')
    return data['K1'], data['K2']


if __name__ == '__main__':
    K1, K2 = load_intrinsics()
    print(K1.shape)
    print(K2.shape)
