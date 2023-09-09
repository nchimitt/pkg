import torch
import math
from ft import ft, ift


def rndsigcorr(corr, N):
    gen_dim = list(range(1,len(corr.shape)+1))
    return ift(torch.sqrt(abs(ft(corr.unsqueeze(0), dim=gen_dim)[0])) * (torch.randn((N, *corr.shape)) + 
                                                     1j*torch.randn((N, *corr.shape))), dim=gen_dim)[0].real


def rndsigpsd(psd, N):
    gen_dim = list(range(1,len(psd.shape)+1))

    return ift(torch.sqrt(psd.unsqueeze(0), dim=gen_dim) * (torch.randn((N, *psd.shape)) + 
                                                     1j*torch.randn((N, *psd.shape))), dim=gen_dim).real

def rnddist(cdf, samps=1):
    print('hi')


def methast(q, samps=1):
    print('hi')


def fftcov(sig1, dim):
    return ift(torch.abs(ft(sig1, dim=dim))**2, dim=dim)

def fftstfn(sig1, dim):
    return ift(torch.abs(ft(sig1, dim=dim))**2, dim=dim)