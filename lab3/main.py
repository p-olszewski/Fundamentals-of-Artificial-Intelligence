import random
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-5 * x))


def fa(weights, inputs):
    return sigmoid(np.dot(inputs, weights))


def train(n):
    p = np.array([[4, 2, -1],
                  [0.01, -1, 3.5],
                  [0.01, 2, 0.01],
                  [-1, 2.5, -2],
                  [-1.5, 2, 1.5]])

    t = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    w = np.array([[0.0629, -0.0805, -0.0685],
                  [0.0812, -0.0443, 0.0941],
                  [-0.0746, 0.0094, 0.0914],
                  [0.0827, 0.0915, -0.0029],
                  [0.0265, 0.0930, 0.0601]])

    for i in range(n):
        nr_example = random.randint(0, 2)
        x = p[:, nr_example]
        y = fa(w, x)
        d = t[:, nr_example] - y
        x_reshape = x.reshape(5, 1)
        d_reshape = d.reshape(3, 1)
        d_w = 0.1 * np.dot(x_reshape, d_reshape.T)
        w += d_w
    return w


if __name__ == '__main__':
    # array features order:
    # number_of_legs, aquatic, flying, feather, oviparous
    cat = np.array([[4, 0.1, 0, 0, 0]])
    weights = train(10)
    print(f"Predict for [4, 0.1, 0, 0, 0]: {fa(weights, cat)}")
