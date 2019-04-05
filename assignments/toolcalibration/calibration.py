import numpy as np


def pivot_calibration(transforms):
    """ Pivot calibration
    Keyword arguments:
    transforms -- A list with 4x4 transformation matrices
    returns -- A vector p_t, which is the offset from any T to the pivot point
    """

    l = len(transforms)  # number of elements in list
    I3 = np.eye(3)  # 3x3 identity matrix

    # initialize empty arrays
    A = np.zeros((3*l, 6))
    b = np.zeros((3*l))

    # fill empty array
    for i in range(0, l):
        T_temp = transforms.pop()
        A[3*i:3*(i+1), 0:3] = T_temp[0:3, 0:3]  # R_i
        A[3*i:3*(i+1), 3:6] = -I3  # -I_3
        b[3*i:3*(i+1)] = -T_temp[:3, 3]  # -p_i

    # make pseudo inverse of A matrix and solve system
    A_pinv = np.linalg.pinv(A)
    x = np.matmul(A_pinv, b)

    # save results
    p_t = x[0:3]
    T = np.eye(4)
    T[:3, 3] = p_t

    return p_t, T
