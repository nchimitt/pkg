import ft
import torch
import numpy as np
import matplotlib.pyplot as plt
import rand
import math


color1 = (0.01, 0.01, 0.01)
color2 = (0.1, 0.5, 0.8)
color3 = (0.99, 0.2, 0.4)
thickness1 = 3
thickness2 = 2
figsize1 = (8,3)

def rgen1():
    T, N = 15, 1001
    sig = 1
    x = ft.linftspace(-T/2, T/2, N)
    y = torch.exp(-x**2/sig)

    z = rand.rndsigcorr(y, N)
    cov = rand.fftcov(z, [0])*0

    for i in range(100):
        z = rand.rndsigcorr(y, N)
        cov += rand.fftcov(z, [0])
    cov /= 100
    plt.figure(figsize=figsize1)
    plt.plot(x, z, color=color1, linewidth=thickness1)
    plt.plot(x, y, color=color2, linewidth=thickness1)
    plt.plot(x, cov, color=color3, linewidth=thickness2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('tex/pix/rnd_gen_1.pdf')


if __name__ == "__main__":
    rgen1()
