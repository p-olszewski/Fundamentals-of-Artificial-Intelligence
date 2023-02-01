import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def adaptation_function(chromosomes):
    return 0.2 * np.sqrt(np.packbits(chromosomes)) + 2.0 * np.sin(2.0 * np.pi * 0.02 * np.packbits(chromosomes)) + 5.0


def genetic_algorithm(chromosomes_amount, pk, pm):
    results = []
    # rand chromosomes population
    random.seed(0)
    chromosomes = [[random.randint(0, 1) for _ in range(8)] for _ in range(chromosomes_amount)]
    length = len(chromosomes)
    # operations
    for generation in range(200):
        # 1. find parent chromosomes
        temp = []
        roulette_wheel = np.add.accumulate(
            (adaptation_function(chromosomes) / sum(adaptation_function(chromosomes))) * 100)
        for i in range(length):
            value = random.uniform(0, 100)
            for j in range(length):
                if roulette_wheel[j] >= value:
                    temp.append(chromosomes[j])
                    break
        chromosomes = temp

        # 2. crossover
        temp = []
        for i in range(0, length - 1, 2):
            first = chromosomes[i]
            second = chromosomes[i + 1]
            if random.random() <= pk:
                cross_point = random.randint(1, len(first) - 1)
                temp.append(first[:cross_point] + second[cross_point:])
                temp.append(second[:cross_point] + first[cross_point:])
            else:
                temp.append(first)
                temp.append(second)
        chromosomes = temp

        # 3. mutation
        for i in range(length):
            if random.random() <= pm:
                mutation_point = random.randint(0, len(chromosomes[i]) - 1)
                if chromosomes[i][mutation_point] == 1:
                    chromosomes[i][mutation_point] = 0
                else:
                    chromosomes[i][mutation_point] = 1
        results.append(np.average(adaptation_function(chromosomes)))
    return results


# pm in title, pk in labels
def show_data(chromosomes_amount, pm_array, pk_array):
    for pm in pm_array:
        for pk in pk_array:
            plt.plot(genetic_algorithm(chromosomes_amount, pk, pm), lw=0.5)
        plt.title("Chromosomes = " + str(chromosomes_amount) + ", PM = " + str(pm))
        legend_labels = ["PK = " + str(pk) for pk in pk_array]
        plt.legend(legend_labels)
        plt.xlabel("Number of generations")
        plt.ylabel("Value of the adaptation function")
        plt.show()


def print_data(chromosomes_amount, pm_array, pk_array):
    data = []
    for pm in pm_array:
        for pk in pk_array:
            result = genetic_algorithm(chromosomes_amount, pk, pm)
            avg_result = round(np.average(result), 3)
            print("Chromosomes=", chromosomes_amount, ", pm =", pm, ", pk =", pk, ", Result =", avg_result)
            data.append((pm, pk, avg_result))
    # save to excel
    df = pd.DataFrame(data, columns=["pm", "pk", "avg_result"])
    pivot = df.pivot_table(index='pm', columns='pk', values='avg_result')
    print(pivot)
    pivot.to_excel("results.xlsx")


if __name__ == '__main__':
    pm_array = [0, 0.01, 0.06, 0.1, 0.2, 0.3, 0.5]
    pk_array = [0.5, 0.6, 0.7, 0.8, 1]
    chromosomes_amount = 50
    # show_data(chromosomes_amount, [0, 0.01, 0.1], pk_array)
    print_data(chromosomes_amount, pm_array, pk_array)
