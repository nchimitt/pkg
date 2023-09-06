import ft
import torch
import numpy
import matplotlib.pyplot as plt


def gauss():
    T, N = 5, 21
    x = ft.linftspace(-T/2, T/2, N)
    y = torch.exp(-x**2/3)

    Y, fs = ft.ft(y, [0], T/N)
    X = ft.linftspace(-fs*N/2, fs*N/2, fs)

    plt.subplot(1,2,1)
    plt.plot(x, y)
    plt.subplot(1,2,2)
    plt.plot(X, Y.real)
    plt.plot(X, Y.image)
    plt.savefig('tex/pix/gauss_ch1_basic.pdf')




if __name__ == "__main__":
    gauss()