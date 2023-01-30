import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def adaptation_function(chromosomes):
    return 0.2 * np.sqrt(np.packbits(chromosomes)) + 2.0 * np.sin(2.0 * np.pi * 0.02 * np.packbits(chromosomes)) + 5.0


def get_parent_population(chromosomes):
    length = len(chromosomes)
    parents = []
    roulette_wheel = np.add.accumulate((adaptation_function(chromosomes) / sum(adaptation_function(chromosomes))) * 100)
    for i in range(length):
        value = random.uniform(0, 100)
        for j in range(length):
            if roulette_wheel[j] >= value:
                parents.append(chromosomes[j])
                break
    return parents


def crossover(parents, pk):
    length = len(parents)
    population = []
    for i in range(0, length - 1, 2):
        first = parents[i]
        second = parents[i + 1]
        if random.random() <= pk:
            cross_point = random.randint(1, len(first) - 1)
            population.append(first[:cross_point] + second[cross_point:])
            population.append(second[:cross_point] + first[cross_point:])
        else:
            population.append(first)
            population.append(second)
    return population


if __name__ == '__main__':
    print("lab5")
