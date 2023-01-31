import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def adaptation_function(chromosomes):
    return 0.2 * np.sqrt(np.packbits(chromosomes)) + 2.0 * np.sin(2.0 * np.pi * 0.02 * np.packbits(chromosomes)) + 5.0


def find_parents(chromosomes):
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


def mutation(chromosomes, pm):
    length = len(chromosomes)
    for i in range(length):
        if random.random() <= pm:
            mutation_point = random.randint(0, len(chromosomes[i]) - 1)
            if chromosomes[i][mutation_point] == 1:
                chromosomes[i][mutation_point] = 0
            else:
                chromosomes[i][mutation_point] = 1
    return chromosomes


# this will be moved to the main function
def genetic_algorithm(pk, pm):
    results = []
    random.seed(0)
    population = [[random.randint(0, 1) for _ in range(8)] for _ in range(200)]  # 200 or 50
    for generation in range(200):
        population = find_parents(population)
        population = crossover(population, pk)
        population = mutation(population, pm)
        results.append(np.average(adaptation_function(population)))
    return results


def show_data(population, pk, pm_array):
    for p in pm_array:
        plt.plot(genetic_algorithm(1, p), lw=0.6)
    plt.title("Population = " + str(population) + ", PK = " + str(pk))
    plt.legend([f'PM = {p}' for p in pm_array])
    plt.xlabel("Number of generations")
    plt.ylabel("Value of the adaptation function")
    plt.show()


if __name__ == '__main__':
    show_data(50, 1, [0, 0.01, 0.06])
    show_data(50, 1, [0.5, 0.7, 1])
    show_data(200, 1, [0, 0.01, 0.2])
    show_data(200, 1, [0.6, 0.8, 1])
