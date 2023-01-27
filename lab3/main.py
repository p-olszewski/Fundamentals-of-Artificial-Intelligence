import random
import numpy as np

BETA_VALUE = 5
LEARNING_RATE = 0.1
EPOCHS = 100


def sigmoid(x):
    return 1 / (1 + np.exp(-BETA_VALUE * x))


def predict(weights, inputs):
    return sigmoid(np.dot(inputs, weights))


def train(weights, examples, outputs):
    for i in range(EPOCHS):
        example = random.randint(0, 2)
        x = examples[:, example]
        y = predict(weights, x)
        d = outputs[:, example] - y
        x_reshape = x.reshape(5, 1)
        d_reshape = d.reshape(3, 1)
        d_w = LEARNING_RATE * np.dot(x_reshape, d_reshape.T)
        weights += d_w
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
    cat = np.array([[4, 0.1, 0, 0, 0]])
    print(f"Predict before for [4, 0.1, 0, 0, 0]: {predict(weights_before_training, cat)}")
    weights_after_training = train(weights_before_training, examples_matrix, requested_outputs)
    print(f"Predict for [4, 0.1, 0, 0, 0]: {predict(weights_after_training, cat)}")
