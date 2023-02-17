import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

RANGE_START = 0
RANGE_END = 9
POINTS = 50
HIDDEN_LAYER_NEURONS = [3, 5, 10, 15, 30, 50]

if __name__ == '__main__':
    # generate x values
    x = np.linspace(RANGE_START, RANGE_END, POINTS)
    size = len(x)

    # CHANGE THIS CODE TO TEST ANOTHER TRAIN TYPE OR FUNCTION
    y1 = 2 * pow(x, 1 / 3) * np.sin(x / 10) * np.cos(3 * x)
    # y1 = 2 * x * np.sin(x)
    train_type = nl.net.train.train_gdx
    ##########################################################

    # normalize the function
    y = y1 / np.abs(max(y1) - min(y1))
    # prepare inputs and targets for training
    inputs = x.reshape(size, 1)
    targets = y.reshape(size, 1)

    print("Type: ", train_type)
    # iterate over the number of neurons in the hidden layer
    for neuron in HIDDEN_LAYER_NEURONS:
        # create a new feedforward neural network
        net = nl.net.newff([[RANGE_START, RANGE_END]], [neuron, 1])
        net.trainf = train_type  # set the train type
        error = net.train(inputs, targets, epochs=1000, show=None, goal=0.00)  # train the network
        prediction = net.sim(inputs)   # predict outputs for x values
        result = mean_squared_error(targets, prediction)  # calculate mean squared error
        print("Hidden layer neurons: ", neuron, " - ", round(result, 4))
        y2 = net.sim(x.reshape(x.size, 1)).reshape(x.size)  # get predicted y values
        plt.plot(x, y2, '-', x, y, '-')  # plot the predicted and real y values
        plt.legend(['learning result', 'real value'])
        plt.title("Function 2, " + str(train_type) + ", " + str(neuron) + " neurons in hidden layer")
        plt.show()
