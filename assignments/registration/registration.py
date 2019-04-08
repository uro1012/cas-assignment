import numpy as np
from scipy.spatial import cKDTree
import math

def paired_points_matching(source, target):
    """
    Calculates the transformation T that maps the source to the target
    :param source: A N x 3 matrix with N 3D points
    :param target: A N x 3 matrix with N 3D points
    :return:
        T: 4x4 transformation matrix mapping source onto target
        R: 3x3 rotation matrix part of T
        t: 1x3 translation vector part of T
    """

    # check source and target dimensions
    assert source.shape == target.shape

    T = np.eye(4)  # empty transform matrix

    # centroid of source (l) and target (r)
    mu_l = [np.mean(source[:, 0]), np.mean(source[:, 1]), np.mean(source[:, 2])]  # source -> l
    mu_r = [np.mean(target[:, 0]), np.mean(target[:, 1]), np.mean(target[:, 2])]  # target -> r

    # compute rotation matrix and translation using SVD
    P = source - mu_l
    Q = target - mu_r
    M = np.dot(np.transpose(P), Q)
    U, W, V_t, = np.linalg.svd(M)
    R = np.dot(np.transpose(V_t), np.transpose(U))
    t = mu_r - np.dot(R, mu_l)

    # fill transformation matrix
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    t = np.reshape(t, (3, 1))  # force 2D array

    return T, R, t


def find_nearest_neighbor(src, dst):
    """
    Finds the nearest neighbor of every point in src in dst
    :param src: A N x 3 point cloud
    :param dst: A N x 3 point cloud
    :return: the
    """

    # get nearest points with KD-tree algorithm
    tree = cKDTree(dst)
    distance, index = tree.query(src)

    return distance, index


def icp(source, target, init_pose=None, max_iterations=30, tolerance=0.0001):
    """
    Iteratively finds the best transformation that mapps the source points onto the target
    :param source: A N x 3 point cloud
    :param target: A N x 3 point cloud
    :param init_pose: A 4 x 4 transformation matrix for the initial pose
    :param max_iterations: default 30
    :param tolerance: maximum allowed error
    :return: A 4 x 4 rigid transformation matrix mapping source to target
            the distances and the error
    """

    # initialize values
    T = init_pose
    distances = 0
    error = 0

    # init pose is the first transformation
    R = init_pose[0:3, 0:3]
    t = init_pose[0:3, 3]
    t = np.reshape(t, (3, 1))

    # iterate up to max iterations
    for count in range(0, max_iterations):
        # move source points according to the last transformation
        source = np.dot(R, np.transpose(source[:])) + t
        source = np.transpose(source)

        # get nearest point
        distance, index = find_nearest_neighbor(source, target)
        # apply PPM algorithm
        T_ppm, R, t = paired_points_matching(source, target[index[:]])
        # combine transformation with last one
        T = np.dot(T_ppm, T)

        # compute error and mean distance
        error_old = error
        error = np.transpose(target[index[:]]) - np.dot(R, np.transpose(source[:])) - t
        error = (error[:]**2).sum()
        distances = np.mean(distance)

        # stop if precise enough
        if abs(error-error_old) < tolerance:
            break

    return T, distances, error


def get_initial_pose(template_points, target_points):
    """
    Calculates an initial rough registration
    (Optionally you can also return a hand picked initial pose)
    :param source:
    :param target:
    :return: A transformation matrix
    """

    T = np.eye(4)

    # X axis rotation
    theta_x = math.pi / 2
    Rx = [[math.cos(theta_x), -math.sin(theta_x)],
          [math.sin(theta_x), math.cos(theta_x)]]

    # XYZ translations
    dx = +600
    dy = -25
    dz = -175
    t = [dx, dy, dz]

    # fill transform matrix T
    T[0:3, 3] = t
    T[1:3, 1:3] = Rx

    return T

