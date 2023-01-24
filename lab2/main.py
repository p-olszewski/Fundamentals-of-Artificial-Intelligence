import numpy as np


class Perceptron:
    def __init__(self):
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

    def train(self, input_array, output_array):
        # train 100 times
        for i in range(100):
            for input_value, output_value in zip(input_array, output_array):
                prediction_value = self.predict(input_value)
                difference = output_value - prediction_value
                if difference == 0:
                    continue
                self.bias += difference
                delta_weight = difference * input_value
                self.weights += delta_weight

    def predict(self, input_array):
        weighted_sum = np.dot(input_array, self.weights) + self.bias
        if weighted_sum > 0:
            return 1
        else:
            return 0


if __name__ == '__main__':
    perceptron = Perceptron()

    # train perceptron
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_outputs = np.array([0, 0, 1, 1])
    perceptron.train(inputs, expected_outputs)

    # test 1
    test_value = np.array([0, 0])
    result_value = perceptron.predict(test_value)
    print("\nTest1\nExpected value: 0\nActual value: " + str(result_value))

    # test 2
    test_value = np.array([1, 0])
    result_value = perceptron.predict(test_value)
    print("\nTest2\nExpected value: 1\nActual value: " + str(result_value))

