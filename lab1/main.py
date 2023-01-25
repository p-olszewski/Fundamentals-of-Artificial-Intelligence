import random
from itertools import permutations
import numpy as np
from numpy import random


# Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum(np.square(np.array(p1) - np.array(p2))))


def cities_init():
    random.seed(1)
    number_of_cities = 0
    while number_of_cities not in range(3, 11):
        try:
            number_of_cities = int(input("Enter the number of cities between 3 and 10: \n"))
        except ValueError:
            print("Invalid input. Please enter a number.")
    result = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(number_of_cities)]
    return result


def brute_force_method(points):
    n = len(points)
    min_dist = float('inf')
    best_path = None

    for path in permutations(points):
        dist = 0
        for i in range(n - 1):
            dist += euclidean_distance(path[i], path[i + 1])
        dist += euclidean_distance(path[-1], path[0])
        if dist < min_dist:
            min_dist = dist
            best_path = path
    print("\nBrute force method path: " + str(best_path) + "\nBrute force method distance: " + str(round(min_dist, 2)))


def n_n_method(points):
    path = [points[0]]  # first city
    remaining_points = points[1:]

    while remaining_points:
        nearest = min(remaining_points, key=lambda x: euclidean_distance(path[-1], x))
        path.append(nearest)
        remaining_points.remove(nearest)

    total_distance = sum(euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1)) + euclidean_distance(
        path[-1], path[0])

    print("\nN-N method path: " + str(path) + "\nN-N method distance: " + str(round(total_distance, 2)))


def randomized_hill_climbing(cities):
    n = len(cities)
    best_path = random.sample(range(n), n)
    best_path_length = sum(euclidean_distance(cities[best_path[i]], cities[best_path[i + 1]]) for i in
                           range(len(best_path) - 1)) + euclidean_distance(cities[best_path[-1]], cities[best_path[0]])

    while True:
        neighbor_path = best_path.copy()
        l1, l2 = random.sample(range(n), 2)
        neighbor_path[l1], neighbor_path[l2] = neighbor_path[l2], neighbor_path[l1]
        neighbor_path_length = sum(euclidean_distance(cities[neighbor_path[i]], cities[neighbor_path[i + 1]]) for i in
                                   range(len(neighbor_path) - 1)) + euclidean_distance(cities[neighbor_path[-1]],
                                                                                       cities[neighbor_path[0]])
        if neighbor_path_length < best_path_length:
            best_path = neighbor_path
            best_path_length = neighbor_path_length
        else:
            print("\nRandomized Hill Climbing method path: " + str([cities[i] for i in best_path]))
            print("Randomized Hill Climbing method distance: " + str(round(best_path_length, 2)))
            return


if __name__ == '__main__':
    citiesArray = cities_init()
    print("\nCities:\n" + str(citiesArray))
    n_n_method(citiesArray)
    brute_force_method(citiesArray)
    randomized_hill_climbing(citiesArray)
