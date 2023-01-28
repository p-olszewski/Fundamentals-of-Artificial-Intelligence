import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt

RANGE_START = 0
RANGE_END = 9


def low_amplitude_function():
    y = 2 * x * np.sin(x)


if __name__ == '__main__':
    x = np.linspace(-7, 7, 30)
    y1 = 2 * x * np.sin(x)
    wsp = np.abs(max(y1) - min(y1))
    y = y1 / wsp
    size = len(x)
    print(x)
    inp = x.reshape(size, 1)
    print(inp)
    tar = y.reshape(size, 1)

    net = nl.net.newff([[-7, 7]], [10, 1])
    net.trainf = nl.net.train.train_gdx
    error = net.train(inp, tar, epochs=1000, show=100, goal=0.00)

    out = net.sim(inp)

    x2 = np.linspace(-6.0, 6.0, 150)
    y2 = net.sim(x2.reshape(x2.size, 1)).reshape(x2.size)
    y3 = out.reshape(size)
    plt.plot(x2, y2, '-', x, y, '.', x, y3, 'p')
    plt.legend(['wynik uczenia', 'wartosc rzeczywista'])
    plt.show()
