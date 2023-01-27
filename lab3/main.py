import random
import numpy as np

BETA_VALUE = 5
LEARNING_RATE = 0.1
EPOCHS = 100


def percent_display(result):
    result_percent = 100 * result
    result_percent = np.around(result_percent, 2)
    print(np.core.defchararray.add(result_percent.astype(str), '%'))


def sigmoid(x):
    return 1 / (1 + np.exp(-BETA_VALUE * x))


def activation(weights, inputs):
    return sigmoid(np.dot(inputs, weights))


def train(weights, examples, outputs):
    for i in range(EPOCHS):
        example = random.randint(0, 2)  # rand an example
        x = examples[:, example]  # from examples
        y = activation(weights, x)  # example for inputs and calculate outputs
        d = outputs[:, example] - y  # output errors
        d_w = LEARNING_RATE * np.dot(x.reshape(5, 1), d.reshape(3, 1).T)  # weight gradient
        weights += d_w  # update weights
    return weights


if __name__ == '__main__':
    examples_matrix = np.array([[4, 2, -1],
                                [0.01, -1, 3.5],
                                [0.01, 2, 0.01],
                                [-1, 2.5, -2],
                                [-1.5, 2, 1.5]])

    requested_outputs = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])

    weights_before_training = np.array([[0.0629, -0.0805, -0.0685],
                                        [0.0812, -0.0443, 0.0941],
                                        [-0.0746, 0.0094, 0.0914],
                                        [0.0827, 0.0915, -0.0029],
                                        [0.0265, 0.0930, 0.0601]])

    # array features order:
    # number_of_legs, aquatic, flying, feather, oviparous
    print("\n[['MAMMAL' 'BIRD' 'FISH']]")

    # mammal
    cat = np.array([[4, 0.1, 0, 0, 0]])
    weights_after_training = train(weights_before_training, examples_matrix, requested_outputs)
    print("\nCat prediction (4, 0.1, 0, 0, 0):")
    percent_display(activation(weights_after_training, cat))

    # bird
    eagle = np.array([[2, 0.1, 0.5, 2, 2]])
    weights_after_training = train(weights_before_training, examples_matrix, requested_outputs)
    print("\nEagle prediction (2, 0.1, 0.5, 2, 2):")
    percent_display(activation(weights_after_training, eagle))

    # fish
    shark = np.array([[0, 2, 0.1, 0.1, 2]])
    weights_after_training = train(weights_before_training, examples_matrix, requested_outputs)
    print("\nShark prediction (0, 2, 0.1, 0.1, 2):")
    percent_display(activation(weights_after_training, shark))
