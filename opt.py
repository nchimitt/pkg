import numpy as np
import scipy
import math
from time import time

def noll_to_zernike(i):
	'''Get the Zernike index from a Noll index.
	
	vectorized version of: https://github.com/ehpor/hcipy/blob/master/hcipy/mode_basis/zernike.py
	Parameters
	----------
	i : int
		The Noll index.
	Returns
	-------
	n : int
		The radial Zernike order.
	m : int
		The azimuthal Zernike order.
	'''
	n = (np.sqrt(2 * i - 1) + 0.5).astype(int) - 1
	m = np.where(n % 2, 
	      2 * ((2 * (i + 1) - n * (n + 1)) // 4).astype(int) - 1,
	      2 * ((2 * i + 1 - n * (n + 1)) // 4).astype(int))
	return n, m * (-1)**(i % 2)


def zernike(coeffs, N=25, sum_z=True):
	'''
    '''
	# Vectorized version
	Nvec, Nz = coeffs.shape
	n,m = noll_to_zernike(np.arange(1, Nz+1))
	sum_ul = int(np.amax(n - np.abs(m))/2)
	k = np.arange(0, sum_ul+1)[:,np.newaxis]
	rad_terms = (-1)**(k) * scipy.special.binom(n[np.newaxis,:] - k, k) * \
                    scipy.special.binom(n[np.newaxis,:] - 2*k, 
					(n[np.newaxis,:]-np.abs(m[np.newaxis,:]))/2 - k)
	rad_terms[np.isnan(rad_terms)] = 0
	sum_mask = np.tile(np.arange(0, sum_ul+1), (Nz, 1)).T
	sum_mask = np.where(sum_mask <= (n - np.abs(m))/2, 1, 0)
	rhox, rhoy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
	rho = np.sqrt(rhox**2 + rhoy**2)
	rad_poly = rho[np.newaxis, np.newaxis,:]**(np.maximum(n[np.newaxis,...] - 2*k, 0))[...,np.newaxis, np.newaxis]
	rad_poly = rad_poly * np.where(rad_poly < 1, 1, 0)
	psi = np.arctan2(rhoy, rhox)
	ang_funs = np.where(m[:,np.newaxis,np.newaxis] < 0, np.sin(m[:,np.newaxis, np.newaxis] * psi[np.newaxis,...]), 
		                        np.cos(m[:,np.newaxis, np.newaxis] * psi[np.newaxis,...]))
	
	Rmn = np.sum((sum_mask * rad_terms)[...,np.newaxis, np.newaxis] * rad_poly, axis=0)
	return np.einsum('ij,jkl->ikl', coeffs, Rmn * ang_funs)