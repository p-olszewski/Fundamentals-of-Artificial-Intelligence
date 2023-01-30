import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def membership_function(binary):
    decimal = np.packbits(binary)
    result = 0.2 * np.sqrt(decimal) + 2.0 * np.sin(2.0 * np.pi * 0.02 * decimal) + 5.0
    return result


if __name__ == '__main__':
    print("lab5")
