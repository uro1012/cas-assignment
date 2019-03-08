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

    # apply threshold
    f = 5  # number of sigmas, experimental factor
    values = neighbour_values(image, seed_point)  # values around seed point
    mu = int(sum(values) / 27)  # mean, 3^3 neighbours including point
    sigma = int(np.std(values))  # standard deviation
    print("mu: ", mu, ", sigma: ", sigma)  # debug
    threshold_mask = threshold2(image, mu-f*sigma, mu+f*sigma)
    print("total number of points in threshold: ", sum(sum(sum(threshold_mask))))  # debug

    # make queue of points around seed point
    queue = enqueue_candidates(threshold_mask, seed_point)
    print("queue size: ", queue.qsize())  # debug

    # segment with connected threshold algorithm
    segmentation_mask = segment(queue, threshold_mask, seed_point)
    print("total number of points in segmentation: ", sum(sum(sum(segmentation_mask))))  # debug

    return segmentation_mask


def threshold2(image, min, max):
    """
    Apply threshold on image with minima and maxima
    :param image: Image to which to apply the threshold
    :param min: Minimal value of the threshold
    :param max: Maximal value of the threshold
    :return: Threshold mask
    """
    threshold_mask = np.zeros(image.shape, np.bool)  # empty mask
    xMax, yMax, zMax = image.shape  # limits of image

    for x in range(0, xMax):
        for y in range(0, yMax):
            for z in range(0, zMax):
                if (image[x, y, z] > min) & (image[x, y, z] < max):
                    threshold_mask[x, y, z] = True

    return threshold_mask


def enqueue_candidates(threshold, point):
    """
    Make a queue of all candidates of the segmentation by growing a cube around the point
    :param threshold: Threshold mask of the image
    :param point: Starting point of the grow
    :return: Queue of all candidates
    """
    xMax, yMax, zMax = threshold.shape  # shape of the mask
    xx, yy, zz = point  # initial point coords
    lim = max(xMax, yMax, zMax)  # limitation of growing
    q = queue.Queue()  # new empty queue

    q.put(point)  # add initial point

    # check every surface by growing a cube around initial point
    for i in range(1, lim):
        x = min(xx+i, xMax-2)
        for y in range(yy-i, yy+i):
            if (y < 0) | (y >= yMax):
                continue
            for z in range(zz-i, zz+i):
                if (z < 0) | (z >= zMax):
                    continue
                p = (x, y, z)
                if threshold[p]:
                    q.put(p)

        x = max(xx-i, 1)
        for y in range(yy-i, yy+i):
            if (y < 0) | (y >= yMax):
                continue
            for z in range(zz-i, zz+i):
                if (z < 0) | (z >= zMax):
                    continue
                p = (x, y, z)
                if threshold[p]:
                    q.put(p)

        y = min(yy+i, yMax-2)
        for x in range(xx-i, xx+i):
            if (x < 0) | (x >= xMax):
                continue
            for z in range(zz-i, zz+i):
                if (z < 0) | (z >= zMax):
                    continue
                p = (x, y, z)
                if threshold[p]:
                    q.put(p)

        y = max(yy-i, 1)
        for x in range(xx-i, xx+i):
            if (x < 0) | (x >= xMax):
                continue
            for z in range(zz-i, zz+i):
                if (z < 0) | (z >= zMax):
                    continue
                p = (x, y, z)
                if threshold[p]:
                    q.put(p)

        z = min(zz+i, zMax-2)
        for x in range(xx-i, xx+i):
            if (x < 0) | (x >= xMax):
                continue
            for y in range(yy-i, yy+i):
                if (y < 0) | (y >= yMax):
                    continue
                p = (x, y, z)
                if threshold[p]:
                    q.put(p)

        z = max(zz-i, 1)
        for x in range(xx-i, xx+i):
            if (x < 0) | (x >= xMax):
                continue
            for y in range(yy-i, yy+i):
                if (y < 0) | (y >= yMax):
                    continue
                p = (x, y, z)
                if threshold[p]:
                    q.put(p)

    return q


def segment(queue, threshold, seed_point):
    """
    Segment all points of the queue
    :param queue: Queue of all candidates of the segmentation
    :param threshold: Threshold of the image
    :param seed_point: Seed point of the segmentation
    :return: Segmentation mask
    """
    mask = np.zeros(threshold.shape, np.bool)  # empty mask
    mask[seed_point] = True  # seed point is part of segment

    # loop through queue and check if neighbours are present
    while not queue.empty():
        p = queue.get()
        mask[p] = check_neighbours(p, mask)

    return mask


def check_neighbours(point, mask):
    """
    Check if there are any neighbours around the given point
    :param point: point to check it's neighbourhood
    :param mask: Segmentation mask with existing points
    :return: True if there is at least one neighbour around the point, false if none
    """
    x, y, z = point  # coords of point
    neighbours = sum(sum(sum(mask[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2])))  # total number of neighbours around point
    if neighbours > 0:
        return True
    else:
        return False


def neighbour_values(image, point):
    """
    Get all values around a given point, including itself
    :param image: The original image where to get the data from
    :param point: The center point of a 3x3 cube
    :return: all values as a list
    """
    i, j, k = point  # coords of point
    # print("i: ", i, ", j: ", j, ", k: ", k)

    values = []  # empty list of values
    for x in range(i - 1, i + 2):
        for y in range(j - 1, j + 2):
            for z in range(k - 1, k + 2):
                values.append(image[x, y, z])

    return values
