import torch
from torch.nn.functional import pad

def ft(signal:torch.Tensor, dim:list, dx=False, zpad=False):
    """The Fourier transform function. Technically a DFT, but with some extra
    stuff to...
    (1) Simplify any fft-shifting
    (2) Generate frequency axis
    (3) Match magnitudes of CFT
    all with just a single function. ft() will handle and size of ft (fft, fft2, ...)

    Args:
        signal (torch.Tensor): _description_
        dim (tuple): _description_
        dx (bool, optional): _description_. Defaults to False.
        zpad (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # extracting which dimensions to take ft, which to not take ft
    ft_dims = torch.as_tensor([signal.shape[x] for x in dim]) # which dimensions to transform
    # temp = list(range((len(signal.shape))))
    # ft_dims = torch.as_tensor([temp[x] for x in dim])
    # sig_dims = torch.arange(len(signal.shape))
    # vals, cnts = torch.unique(torch.sort(torch.cat((sig_dims, ft_dims)))[0], return_counts=True)
    # non_ft_dims = vals[cnts < 2] # which dimensions to not transform

    # if no sample spacing given, do standard fft
    if type(dx) == bool: # maybe i'll make this better in the future
        # standard fft call
        return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(signal, dim=dim), dim=dim), dim=dim
                                  ) / torch.prod(torch.as_tensor([signal.shape[x] for x in dim]), 0)
    else: # if sample spacing given, do fft along with returning frequency grid
        fs = 1/(ft_dims * dx) # inverse of sample rate
        return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(signal, dim=dim), dim=dim), dim=dim
        ) / torch.prod(ft_dims, 0), fs
    


def ift(ft_signal:torch.Tensor, dim:list, df=False, zpad=0):
    """The inverse Fourier transform function. Technically a DFT, but with some extra
    stuff to :
    (1) Simplify any fft-shifting
    (2) Generate frequency axis
    (3) 

    Args:
        ft_signal (torch.Tensor): _description_
        dim (tuple): _description_
        dx (bool, optional): _description_. Defaults to False.
        zpad (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # extracting which dimensions to take ft, which to not take ft
    ift_dims = torch.as_tensor([ft_signal.shape[x] for x in dim]) # which dimensions to transform
    # sig_dims = torch.arange(len(ft_signal.shape))
    # vals, cnts = torch.unique(torch.sort(torch.cat((sig_dims, ft_dims)))[0], return_counts=True)
    # non_ft_dims = vals[cnts < 2] # which dimensions to not transform

    # if no sample spacing given, do standard ifft
    if type(df) == bool: # maybe i'll make this better in the future
        # standard fft call
        return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(ft_signal, dim=dim), dim=dim), dim=dim
                                  ) * torch.prod(torch.as_tensor([ft_signal.shape[x] for x in dim]), 0)
    else: # if sample spacing given, do fft along with returning frequency grid
        fs = 1/(ift_dims * df) # inverse of sample rate
        return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(ft_signal, dim=dim), dim=dim), dim=dim
        ) * torch.prod(ift_dims, 0), fs
    
def linftspace(T, N):
    if N % 2: #odd
        return torch.linspace(-T/2, T/2, N)
    else:
        return torch.linspace(-T/2, T/2 - T/N, N)