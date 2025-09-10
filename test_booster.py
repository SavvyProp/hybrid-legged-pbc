import numpy as np
import matplotlib.pyplot as plt

for c in range(10):
    x = np.linspace(0, 10, 100)
    y = np.sin(x + c / 2.0) + c
    plt.plot(x, y)
    plt.savefig("a{}.png".format(c))
    plt.clf()