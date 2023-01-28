import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

RANGE_START = 0
RANGE_END = 9
POINTS = 50
HIDDEN_LAYER_NEURONS = [3, 5, 10, 15, 30, 50]

if __name__ == '__main__':
    x = np.linspace(RANGE_START, RANGE_END, POINTS)
    size = len(x)

    # CHANGE THIS CODE TO TEST ANOTHER TRAIN TYPE OR FUNCTION
    y1 = 2 * pow(x, 1 / 3) * np.sin(x / 10) * np.cos(3 * x)
    # y1 = 2 * x * np.sin(x)
    train_type = nl.net.train.train_gdx
    ##########################################################

    y = y1 / np.abs(max(y1) - min(y1))
    inputs = x.reshape(size, 1)
    targets = y.reshape(size, 1)

    print("Type: ", train_type)
    for neuron in HIDDEN_LAYER_NEURONS:
        net = nl.net.newff([[RANGE_START, RANGE_END]], [neuron, 1])
        net.trainf = train_type
        error = net.train(inputs, targets, epochs=1000, show=None, goal=0.00)
        prediction = net.sim(inputs)
        difference = mean_squared_error(targets, prediction)
        print("Hidden layer neurons: ", neuron, " - ", round(difference, 4))
        y2 = net.sim(x.reshape(x.size, 1)).reshape(x.size)
        plt.plot(x, y2, '-', x, y, '-')
        plt.legend(['learning result', 'real value'])
        plt.title("Function 2, " + str(train_type) + ", " + str(neuron) + " neurons in hidden layer")
        plt.show()
