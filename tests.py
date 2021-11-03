import math
import random
import numpy as np

from utils import get_non_dominated


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


def generate_circle_set(radius=5, nd_points=1000, total_points=10000, x_center=0, y_center=0):
    """
    This function generates a set with a circle boundary that has a known pareto front.
    :param radius: The radius of the circle
    :param nd_points: The number of points in the non-dominated set.
    :param total_points: The total number of points for the set.
    :return: The non-dominated set and the complete set.
    """
    nd_list = []
    complete_list = []
    dominated_points = total_points - nd_points

    for _ in range(nd_points):  # Generate the non dominated points in the shape of a circle.
        theta = np.random.uniform(low=0, high=math.pi/2)  # A random angle.
        x = x_center + radius * math.cos(theta)  # The x coordinate.
        y = y_center + radius * math.sin(theta)  # The y coordinate.
        coord = tuple([x, y])
        nd_list.append(coord)
        complete_list.append(coord)

    for point in random.choices(complete_list, k=dominated_points):  # Generate the dominated points.
        new_x = point[0] - min(1e-10, random.random())
        new_y = point[1] - min(1e-10, random.random())
        coord = tuple([new_x, new_y])
        complete_list.append(coord)

    nd_set = set(nd_list)
    complete_set = set(complete_list)
    return nd_set, complete_set


def generate_arbitrary_set(nd_points=1000, total_points=10000, max_x=5, max_y=5):
    """
    This function generates a set with an arbitrary shape.
    :param nd_points: The total number of non-dominated points.
    :param total_points: The total number of points in the set.
    :return: The non-dominated set and the complete set.
    """
    circle_nd_points = math.ceil(nd_points / 2)  # We generate half the points
    circle_points = math.ceil(total_points / 2)
    dominated_points = total_points - circle_points
    nd_set, complete_set = generate_circle_set(radius=max_y, nd_points=circle_nd_points, total_points=circle_points)

    nd_list = list(nd_set)
    complete_list = list(complete_set)
    nd_list.sort(key=lambda x: (-x[0], x[1]))  # Sort on x coordinate first and then y coordinate.

    for i in range(len(nd_list)-1):
        coord1 = nd_list[i]
        coord2 = nd_list[i+1]

        new_x = np.random.uniform(low=coord1[0], high=coord2[0])  # A random new x coordinate between the two points.
        new_y = np.random.uniform(low=coord2[1], high=coord1[1])  # Y for second coordinate is lower, so reverse.
        coord = tuple([new_x, new_y])

        nd_list.append(coord)
        complete_list.append(coord)

    if nd_points % 2 == 0:  # This brings us to the required number of points when this is even.
        coord = tuple([max_x, 0])
        nd_list.append(coord)
        complete_list.append(coord)

    for point in random.choices(complete_list, k=dominated_points):  # Generate the dominated points.
        new_x = point[0] - min(1e-10, random.random())
        new_y = point[1] - min(1e-10, random.random())
        coord = tuple([new_x, new_y])
        complete_list.append(coord)

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


if __name__ == '__main__':
    test_get_non_dominated()
