import math
from collections import defaultdict
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


def held_karp(points):
    n = len(points)
    # tworzenie macierzy odległości między punktami
    dist = [[math.inf for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            dist[i][j] = dist[j][i] = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
    # tworzenie tablicy dynamicznej
    dp = defaultdict(lambda: defaultdict(lambda: math.inf))
    for i in range(n):
        dp[1 << i][i] = 0
    for mask in range(1, 1 << n):
        for i in range(n):
            if mask & (1 << i) == 0:
                continue
            for j in range(n):
                if i != j and mask & (1 << j) != 0:
                    dp[mask][i] = min(dp[mask][i], dp[mask ^ (1 << i)][j] + dist[j][i])
    # znajdowanie najkrótszej ścieżki
    path = []
    mask = (1 << n) - 1
    i = 0
    for _ in range(n):
        path.append(i)
        min_dist = math.inf
        next_i = 0
        for j in range(n):
            if i != j and mask & (1 << j) != 0 and dp[mask][i] + dist[i][j] - dp[mask ^ (1 << j)][j] < min_dist:
                min_dist = dp[mask][i] + dist[i][j] - dp[mask ^ (1 << j)][j]
                next_i = j
        mask = mask ^ (1 << i)
        i = next_i
    path.append(0)
    print("\nHeld-Karp method path: " + str(path) + "\nHeld-Karp method distance: " + str(round(min_dist, 2)))


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


if __name__ == '__main__':
    citiesArray = cities_init()
    print("\nCities:\n" + str(citiesArray))
    n_n_method(citiesArray)
    brute_force_method(citiesArray)
    held_karp(citiesArray)
