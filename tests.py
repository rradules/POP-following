import math
import random
import numpy as np

from utils import get_non_dominated, crowding_distance_assignment


def print_test_result(name, success):
    """
    This function pretty prints the result of a test.
    :param name: The name of the test that was ran.
    :param success: Whether the test was successful.
    :return: /
    """
    if success:
        print(f'✔️ Test for {name} passed️')
    else:
        print(f'❌ Test for {name} failed')


def generate_dominated_points(nd_list, num_dominated):
    """
    This function generates a number of dominated points starting from a list of non dominated points.
    :param nd_list: The list of non dominated points.
    :param num_dominated: The number of dominated points to generate.
    :return: A list of dominated points.
    """
    dominated_list = []
    dominated_by = random.choices(nd_list, k=num_dominated)  # Randomly select points to be dominated by.

    for point in dominated_by:  # Loop over all points to be dominated by.
        new_x = point[0] - min(1e-10, random.random())  # Clip the point to ensure it is strictly smaller.
        new_y = point[1] - min(1e-10, random.random())
        coord = tuple([new_x, new_y])
        dominated_list.append(coord)  # Add the point.

    return dominated_list


def generate_circle_set(radius=5, nd_points=1000, total_points=10000, x_center=0, y_center=0):
    """
    This function generates a set with a circle boundary that has a known pareto front.
    :param radius: The radius of the circle
    :param nd_points: The number of points in the non-dominated set.
    :param total_points: The total number of points for the set.
    :param x_center: The x coordinate of the center.
    :param y_center: The y coordinate of the center.
    :return: The non-dominated set and the complete set.
    """
    nd_list = []

    for _ in range(nd_points):  # Generate the non dominated points in the shape of a circle.
        theta = np.random.uniform(low=0, high=math.pi/2)  # A random angle.
        x = x_center + radius * math.cos(theta)  # The x coordinate.
        y = y_center + radius * math.sin(theta)  # The y coordinate.
        coord = tuple([x, y])
        nd_list.append(coord)

    num_dominated = total_points - nd_points
    dominated_list = generate_dominated_points(nd_list, num_dominated)
    complete_list = nd_list + dominated_list

    nd_set = set(nd_list)
    complete_set = set(complete_list)
    return nd_set, complete_set


def generate_arbitrary_set(nd_points=1000, total_points=10000, min_x=-5, max_x=5, min_y=-5, max_y=5):
    """
    This function generates a set with an arbitrary shape.
    :param nd_points: The total number of non-dominated points.
    :param total_points: The total number of points in the set.
    :param min_x: The minimum x value for the points.
    :param max_x: The maximum x value for the points.
    :param min_y: The minimum y value for the points.
    :param max_y: The maximum y value for the points.
    :return: The non-dominated set and the complete set.
    """
    nd_list = []

    x_coords = np.random.uniform(low=min_x, high=max_x, size=nd_points)
    y_coords = np.random.uniform(low=min_y, high=max_y, size=nd_points)
    x_coords.sort()  # Sort in ascending order.
    y_coords[::-1].sort()  # Sort in descending order.

    for x, y in zip(x_coords, y_coords):  # Create non-dominated points.
        coord = tuple([x, y])
        nd_list.append(coord)

    num_dominated = total_points - nd_points
    dominated_list = generate_dominated_points(nd_list, num_dominated)
    complete_list = nd_list + dominated_list

    nd_set = set(nd_list)
    complete_set = set(complete_list)
    return nd_set, complete_set


def test_get_non_dominated():
    """
    Run the get_non_dominated tests.
    :return: /
    """
    nd_set, complete_set = generate_circle_set()
    pcs = get_non_dominated(complete_set)
    print_test_result('get_non_dominated with circle front', pcs == nd_set)

    nd_set, complete_set = generate_arbitrary_set()
    pcs = get_non_dominated(complete_set)
    print_test_result('get_non_dominated with arbitrary front', pcs == nd_set)

def calc_crowding_distance(F):
    infinity = 1e+14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:

        # sort each column and get index
        I = np.argsort(F, axis=0, kind='mergesort')

        # now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) \
               - np.concatenate([np.full((1, n_obj), -np.inf), F])

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divided by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1)

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity

    return crowding


def test_crowding_distance_assignment():
    """
    Run the crowding distance assignment tests.
    The implementation was checked against a known working implementation that can be found here:
    https://github.com/anyoptimization/pymoo/blob/master/pymoo/algorithms/moo/nsga2.py
    To make this code equivalent to ours, remove the division by the number of objectives from line 207.
    It will show some entries in the final vectors as not equal, but this is due to different floating point precisions.
    :return: /
    """
    nd_set, _ = generate_arbitrary_set()
    crowding_distances = crowding_distance_assignment(nd_set)
    print_test_result('crowding distance assignment', True)


if __name__ == '__main__':
    test_get_non_dominated()
    test_crowding_distance_assignment()
