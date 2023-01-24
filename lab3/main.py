import numpy as np


def train_single_layer_network(W, P, T, n):
    for i in range(n):
        # Obliczenie wyjścia sieci dla każdego przykładu
        Y = sigmoid(np.dot(W, P))

        # Obliczenie błędu sieci dla każdego przykładu
        E = T - Y

        # Obliczenie gradientu błędu dla każdego przykładu
        gradient = np.dot(E, P.T)

        # Aktualizacja wag sieci zgodnie z gradientem błędu
        # W += learning_rate * gradient

    return W


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    P = np.array([[4.0, 2.0, -1.0], [0.01, -1.0, 3.5], [0.01, 2.0, 0.01], [-1.0, 2.5, -2.0], [-1.5, 2.0, 1.5]])
    T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
