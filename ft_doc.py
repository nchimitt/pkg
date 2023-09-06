import ft
import torch
import numpy as np
import matplotlib.pyplot as plt


color1 = (0.01, 0.01, 0.01)
color2 = (0.1, 0.5, 0.8)
color3 = (0.99, 0.2, 0.4)
thickness1 = 3
thickness2 = 3
figsize1 = (8,3)

def gauss_ft():
    T, N = 15, 200
    sig = 0.2
    x = ft.linftspace(-T/2, T/2, N)
    y = torch.exp(-x**2/sig)

    Y, fs = ft.ft(y, [0], T/N)
    X = ft.linftspace(-fs*N/2, fs*N/2, N)

    Y_analytical = np.sqrt(np.pi * sig) * np.exp(-(np.pi*X)**2*sig)

    plt.figure(figsize=figsize1)
    plt.subplot(1,2,1)
    plt.plot(x, y, color=color1, linewidth=thickness1)
    plt.subplot(1,2,2)
    plt.plot(X, Y.real, color=color1, linewidth=thickness1, label='Real')
    plt.plot(X, Y.imag, color=color2, linewidth=thickness1, linestyle='--', label='Imaginary')
    plt.plot(X, Y_analytical, color=color3, linewidth=thickness2, linestyle=(0, (5, 10)), label='Analytical')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tex/pix/gauss_ch1_basic.pdf')


def gauss_ift():
    T, N = 15, 101
    sig = 0.1
    X = ft.linftspace(-T/2, T/2, N)
    Y_analytical = np.sqrt(np.pi * sig) * np.exp(-(np.pi*X)**2*sig)
    

    y, fs = ft.ift(Y_analytical, [0], T/N)
    x = ft.linftspace(-fs*N/2, fs*N/2, N)

    y_analytical = torch.exp(-x**2/sig)

    plt.figure(figsize=figsize1)
    plt.subplot(1,2,1)
    plt.plot(X, Y_analytical, color=color1, linewidth=thickness1)
    plt.subplot(1,2,2)
    plt.plot(x, y.real, color=color1, linewidth=thickness1, label='Real')
    plt.plot(x, y.imag, color=color2, linewidth=thickness1, linestyle='--', label='Imaginary')
    plt.plot(x, y_analytical, color=color3, linewidth=thickness2, linestyle=(0, (5, 10)), label='Analytical')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tex/pix/gauss_ift_ch1_basic.pdf')


# def dft_linspace_issues():




if __name__ == "__main__":
    gauss_ft()
    gauss_ift()