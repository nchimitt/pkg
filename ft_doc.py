import ft
import torch
import numpy as np
import matplotlib.pyplot as plt


color1 = (0.01, 0.01, 0.01)
color2 = (0.1, 0.5, 0.8)
color3 = (0.99, 0.2, 0.4)
thickness1 = 3
thickness2 = 2
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
    T, N = 15, 200
    sig = 0.2
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
def ft_table():
    T1, N1, a = 5, 1001, 5
    t1 = ft.linftspace(-T1/2, T1/2, N1)
    s1 = torch.exp(-a*t1)
    s1[t1 < 0] = 0
    S1, fs1 = ft.ft(s1, [0], T1/N1)
    f1 = ft.linftspace(-fs1*N1/2, fs1*N1/2, N1)
    S1_a = 1 / (a + 1j*2*np.pi*f1)
    plt.figure(figsize=figsize1)
    plt.subplot(1,2,1)
    plt.plot(t1, s1, color=color1, linewidth=thickness1)
    plt.subplot(1,2,2)
    plt.plot(f1, S1.real, color=color1, linewidth=thickness1, label='Real')
    plt.plot(f1, S1.imag, color=color2, linewidth=thickness1, label='Imag')
    plt.plot(f1, S1_a.real, color=color3, linewidth=thickness2, linestyle=(0, (5, 10)), label='Real Analytical')
    plt.plot(f1, S1_a.imag, color=color3, linewidth=thickness2, linestyle=(0, (5, 10)), label='Imag Analytical')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tex/pix/ft_table_1.pdf')
    
    T1, N1, a = 4, 501, 15
    t1 = ft.linftspace(-T1/2, T1/2, N1)
    s1 = t1*torch.exp(-a*t1)
    s1[t1 < 0] = 0
    S1, fs1 = ft.ft(s1, [0], T1/N1)
    f1 = ft.linftspace(-fs1*N1/2, fs1*N1/2, N1)
    S1_a = 1 / (a + 1j*2*np.pi*f1)**2
    plt.figure(figsize=figsize1)
    plt.subplot(1,2,1)
    plt.plot(t1, s1, color=color1, linewidth=thickness1)
    plt.subplot(1,2,2)
    plt.plot(f1, S1.real, color=color1, linewidth=thickness1, label='Real')
    plt.plot(f1, S1.imag, color=color2, linewidth=thickness1, label='Imag')
    plt.plot(f1, S1_a.real, color=color3, linewidth=thickness2, linestyle=(0, (5, 10)), label='Real Analytical')
    plt.plot(f1, S1_a.imag, color=color3, linewidth=thickness2, linestyle=(0, (5, 10)), label='Imag Analytical')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tex/pix/ft_table_2.pdf')

    T1, N1, a, f = 4, 101, 20, 10
    t1 = ft.linftspace(-T1/2, T1/2, N1)
    s1 = torch.sinc(f*t1)
    # s1[t1 < 0] = 0
    S1, fs1 = ft.ft(s1, [0], T1/N1)
    f1 = ft.linftspace(-fs1*N1/2, fs1*N1/2, N1)
    S1_a = f1*0
    S1_a[np.abs(f1/(f/2)) < 1] = 1 / f
    plt.figure(figsize=figsize1)
    plt.subplot(1,2,1)
    plt.plot(t1, s1, color=color1, linewidth=thickness1)
    plt.subplot(1,2,2)
    plt.plot(f1, S1.real, color=color1, linewidth=thickness1, label='Real')
    plt.plot(f1, S1.imag, color=color2, linewidth=thickness1, label='Imag')
    plt.plot(f1, S1_a.real, color=color3, linewidth=thickness2, label='Real Analytical')
    plt.plot(f1, S1_a.real, color=color3, linewidth=thickness2, label='Imag Analytical')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tex/pix/ft_table_3.pdf')

    T1, N1, a, f = 2, 101, 20, 3
    t1 = ft.linftspace(-T1/2, T1/2, N1)
    s1 = torch.cos(2*np.pi*f*t1)
    # s1[t1 < 0] = 0
    S1, fs1 = ft.ft(s1, [0], T1/N1)
    f1 = ft.linftspace(-fs1*N1/2, fs1*N1/2, N1)
    S1_a = f1*0
    S1_a[torch.argmin(abs(f1 - f))] = 0.5
    S1_a[torch.argmin(abs(f1 + f))] = 0.5
    plt.figure(figsize=figsize1)
    plt.subplot(1,2,1)
    plt.plot(t1, s1, color=color1, linewidth=thickness1)
    plt.subplot(1,2,2)
    plt.plot(f1, S1.real, color=color1, linewidth=thickness1, label='Real')
    plt.plot(f1, S1.imag, color=color2, linewidth=thickness1, label='Imag')
    plt.plot(f1, S1_a.real, color=color3, linewidth=thickness2, label='Real Analytical')
    plt.plot(f1, S1_a.real, color=color3, linewidth=thickness2, label='Imag Analytical')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tex/pix/ft_table_4.pdf')

    print(fs1*torch.amax(S1.real))


if __name__ == "__main__":
    gauss_ft()
    gauss_ift()
    ft_table()