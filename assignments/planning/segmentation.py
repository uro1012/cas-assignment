import numpy as np
from scipy import ndimage
import queue


def region_grow(image, seed_point):
    """
    Performs a region growing on the image from seed_point
    :param image: An 3D grayscale input image
    :param seed_point: The seed point for the algorithm
    :return: A 3D binary segmentation mask with the same dimensions as image
    """
    segmentation_mask = np.zeros(image.shape, np.bool)
    segmentation_mask[seed_point] = True  # seed point is part of segment

    f = 3  # factor
    iMax, jMax, kMax = image.shape  # image size
    i, j, k = seed_point  # seek point coords
    original_value = image[seed_point]  # value at seed point

    # apply threshold
    tol = 250  # threshold tolerance
    threshold_mask = threshold2(image, original_value-tol, original_value+tol)
    print("total number of points in threshold: ", sum(sum(sum(threshold_mask))))

    # look for parts belonging to segment, starting from seed point
    for xx in range(0, iMax):
        x = max(i - xx, 2)
        for yy in range(0, jMax):
            y = max(j - yy, 2)
            for zz in range(0, kMax):
                z = max(k - zz, 2)
                if not threshold_mask[x, y, z]:
                    continue
                else:
                    point = [x, y, z]
                    segmentation_mask = check_neighbours(image, point, segmentation_mask, f)

                z = min(k + zz, kMax-2)
                if not threshold_mask[x, y, z]:
                    continue
                else:
                    point = [x, y, z]
                    segmentation_mask = check_neighbours(image, point, segmentation_mask, f)

            y = min(j + yy, jMax-2)
            for zz in range(0, kMax):
                z = max(k - zz, 2)
                if not threshold_mask[x, y, z]:
                    continue
                else:
                    point = [x, y, z]
                    segmentation_mask = check_neighbours(image, point, segmentation_mask, f)

                z = min(k + zz, kMax-2)
                if not threshold_mask[x, y, z]:
                    continue
                    point = [x, y, z]
                    segmentation_mask = check_neighbours(image, point, segmentation_mask, f)

        x = min(i + xx, iMax-2)
        for yy in range(0, jMax):
            y = max(j - yy, 2)
            for zz in range(0, kMax):
                z = max(k - zz, 2)
                if not threshold_mask[x, y, z]:
                    continue
                else:
                    point = [x, y, z]
                    segmentation_mask = check_neighbours(image, point, segmentation_mask, f)

                z = min(k + zz, kMax-2)
                if not threshold_mask[x, y, z]:
                    continue
                else:
                    point = [x, y, z]
                    segmentation_mask = check_neighbours(image, point, segmentation_mask, f)

            y = min(j + yy, jMax-2)
            for zz in range(0, kMax):
                z = max(k - zz, 2)
                if not threshold_mask[x, y, z]:
                    continue
                else:
                    point = [x, y, z]
                    segmentation_mask = check_neighbours(image, point, segmentation_mask, f)

                z = min(k + zz, kMax-2)
                if not threshold_mask[x, y, z]:
                    continue
                else:
                    point = [x, y, z]
                    segmentation_mask = check_neighbours(image, point, segmentation_mask, f)

    print("total number of points in segmentation: ", sum(sum(sum(segmentation_mask))))

    return segmentation_mask


def threshold2(image, min, max):

    threshold_mask = np.zeros(image.shape, np.bool)
    iMax, jMax, kMax = image.shape

    for i in range(0, iMax):
        for j in range(0, jMax):
            for k in range(0, kMax):
                if (image[i, j, k] > min) & (image[i, j, k] < max):
                    threshold_mask[i, j, k] = True

    return threshold_mask


def check_neighbours(image, point, mask, f):

    x, y, z = point
    neighbours = sum(sum(sum(mask[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2])))
    values = neighbour_values(image, point)
    mu = sum(values) / 27  # 3^3 neighbours including point
    sigma = np.std(values)
    if (image[x, y, z] < mu + f * sigma) & (image[x, y, z] > mu - f * sigma) & (neighbours > 0):
        mask[x, y, z] = True

    return mask


def neighbour_values(image, point):

    i, j, k = point
    # print("i: ", i, ", j: ", j, ", k: ", k)

    values = []
    for x in range(i - 1, i + 2):
        for y in range(j - 1, j + 2):
            for z in range(k - 1, k + 2):
                values.append(image[x, y, z])
                # print("v: ", image[x, y, z], ", x: ", x, ", y: ", y, ", z: ", z)

    return values
