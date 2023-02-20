import matplotlib.pyplot as plt
import numpy as np
import helper as hlp

from mpl_toolkits.mplot3d import Axes3D
from submission import *
from loader import *


def main():
    # 1. Load the two temple images and the points from data/some_corresp.npz
    img1 = cv.imread('data/im1.png')
    img2 = cv.imread('data/im2.png')
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    pts1, pts2, n = load_correspond()

    # 2. Run eight_point to compute F
    F = eight_points_algorithm(pts1, pts2, img1.shape, img2.shape)
    # hlp.displayEpipolarF(img1, img2, F)

    # 3. Load points in image 1 from data/temple_coords.npz
    pts1 = load_template()

    # 4. Run epipolar_correspondences to get points in image 2
    pts2 = epipolar_correspondences(img1, img2, F, pts1)

    gs = plt.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.axis('off')
    ax2.axis('off')

    for point in pts1:
        cv.circle(img1, (point[0], point[1]), 3, (0, 0, 255), 3)
    for point in pts2:
        cv.circle(img2, (point[0], point[1]), 3, (0, 0, 255), 3)
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()
    """
    # 测试匹配效果
    hlp.epipolarMatchGUI(img1, img2, F)
    gs = plt.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.axis('off')
    ax2.axis('off')

    for point in pts1:
        cv.circle(img1, (point[0], point[1]), 3, (0, 0, 255), 3)
    for point in pts2:
        cv.circle(img2, (point[0], point[1]), 3, (0, 0, 255), 3)
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()
    """

    # 5. Compute the camera projection matrix P1
    K1, K2 = load_intrinsics()
    P1 = np.zeros((3, 4), dtype=float64)
    row, col = ([0, 1, 2], [0, 1, 2])
    P1[row, col] = 1
    P1 = K1.dot(P1)

    # 6. Use camera2 to get 4 camera projection matrices P2
    E = essential_matrix(F, K1, K2)
    P = hlp.camera2(E)

    # 7. Run triangulate using the projection matrices
    # 8. Figure out the correct P2
    res = -1
    inx = -1
    for i in range(0, 4):
        P2 = P[:, :, i]
        P2 = K2.dot(P2)
        pts3d = triangulate(P1, pts1, P2, pts2)

        count = 0
        for j in pts3d:
            if j[2] > 0:
                count += 1
        print('z > 0: ', count / len(pts3d))

        if count > res:
            res = count
            inx = i

    # 9. Scatter plot the correct 3D points
    P2 = P[:, :, inx]
    P2 = K2.dot(P2)
    pts3d = triangulate(P1, pts1, P2, pts2)

    # 逆向投影的误差计算
    pts3d_h = np.concatenate([pts3d, np.ones((len(pts3d), 1))], axis=1)
    re = P1.dot(pts3d_h.T)
    re = re[:2, :] / re[2, :]
    print('逆向投影平均误差', np.average(np.sum(np.abs(re - pts1.T) ** 0.5, axis=0)))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])

    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()

    # 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
    R1 = np.identity(3)
    t1 = np.zeros((3, 1))
    R2 = P[:, :, inx][:, :3]
    t2 = P[:, :, inx][:, 3]
    np.savez('data/extrinsics.npz', R1=R1, R2=R2, t1=t1, t2=t2)


if __name__ == "__main__":
    main()
