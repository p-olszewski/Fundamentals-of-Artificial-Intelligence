import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def adaptation_function(chromosomes):
    return 0.2 * np.sqrt(np.packbits(chromosomes)) + 2.0 * np.sin(2.0 * np.pi * 0.02 * np.packbits(chromosomes)) + 5.0


def get_roulette_wheel(population):
    return np.add.accumulate((adaptation_function(population) / sum(adaptation_function(population))) * 100)


if __name__ == '__main__':
    print("lab5")
