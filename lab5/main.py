import random
import matplotlib.pyplot as plt
import numpy as np


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


# pk in title, pm in labels
# def show_data(population, pk_array, pm_array):
#     for pk in pk_array:
#         for pm in pm_array:
#             plt.plot(genetic_algorithm(pk, pm))
#         plt.title("Population = " + str(population) + ", PK = " + str(pk))
#         legend_labels = ["PM = " + str(pm) for pm in pm_array]
#         plt.legend(legend_labels)
#         plt.xlabel("Number of generations")
#         plt.ylabel("Value of the adaptation function")
#         plt.show()


# pm in title, pk in labels
def show_data(population, pk_array, pm_array):
    for pm in pm_array:
        for pk in pk_array:
            plt.plot(genetic_algorithm(pk, pm), lw=0.5)
        plt.title("Population = " + str(population) + ", PM = " + str(pm))
        legend_labels = ["PK = " + str(pk) for pk in pk_array]
        plt.legend(legend_labels)
        plt.xlabel("Number of generations")
        plt.ylabel("Value of the adaptation function")
        plt.show()


def print_data(pm_array, pk_array):
    for pm in pm_array:
        for pk in pk_array:
            result = genetic_algorithm(pk, pm)
            avg_result = round(np.average(result), 2)
            print("Population 200, pm =", pm, ", pk =", pk, ", Result =", avg_result)


if __name__ == '__main__':
    pm_array = [0, 0.01, 0.06, 0.1, 0.2, 0.3, 0.5]
    pk_array = [0.5, 0.6, 0.7, 0.8, 1]
    show_data(50, pk_array, [0, 0.01, 0.06])
    show_data(200, pk_array, [0, 0.01, 0.2])
    print_data(pm_array, pk_array)
