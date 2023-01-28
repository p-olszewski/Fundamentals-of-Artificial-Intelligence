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
    y1 = 2 * x * np.sin(x)
    y = y1 / np.abs(max(y1) - min(y1))
    inputs = x.reshape(size, 1)
    targets = y.reshape(size, 1)

    net = nl.net.newff([[RANGE_START, RANGE_END]], [10, 1])
    net.trainf = nl.net.train.train_gdx
    error = net.train(inputs, targets, epochs=1000, show=None, goal=0.00)
    prediction = net.sim(inputs)
    difference = mean_squared_error(targets, prediction)
    print(round(difference, 4))
    print(difference)

    x2 = np.linspace(RANGE_START, RANGE_END, POINTS)
    y2 = net.sim(x2.reshape(x2.size, 1)).reshape(x2.size)
    plt.plot(x2, y2, '-', x, y, '-')
    plt.legend(['learning result', 'real value'])
    plt.show()
