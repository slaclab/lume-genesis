import numpy as np
from scipy.optimize import curve_fit
from scipy.special import *
from genesis import parsers


def load_slices(fname):

    gout = parsers.parse_genesis_out(fname)
    slices = gout['slice_data']
    zsep = gout['input_parameters']['zsep']
    xlamds = gout['input_parameters']['xlamds'] 
    Nslice = len(slices)
   
    return slices, zsep, Nslice, xlamds


def power_spectrum(slices):

    Z0 = 120 * np.pi
    power = np.asarray([s['data']['p_mid'][-1] for s in slices])
    phi_mid = np.asarray([s['data']['phi_mid'][-1] for s in slices])
    field = np.sqrt(power) * np.exp(1j*phi_mid)
    power_fft = np.abs(np.fft.fftshift(np.fft.fft(field)))**2

    return power_fft


def freq_domain_eV(zsep,Nslice,xlamds):

    #constants
    hbar = 6.582e-16 #in eV
    c = 2.997925e8

    #omega of the radiation in eV
    omega = hbar * 2.0 * np.pi / (xlamds/c);
    df = hbar * 2.0 * np.pi/Nslice/zsep/(xlamds/c);

    freq = np.linspace(omega - Nslice/2 * df, omega + Nslice/2 * df,Nslice)

    return freq


def gaussian(x, *p):

    A, mu, sigma, bg = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2)) + bg

def FWHM(power_fft,omega):

    power_fft = power_fft / np.max(power_fft)
    peak_pos = np.argmax(power_fft)
    p0 = [1.0, omega[peak_pos], 0.15, 1e-4]
    window = 10
    coeff, var_matrix = curve_fit(gaussian, omega[peak_pos-window:peak_pos+window], power_fft[peak_pos-window:peak_pos+window], p0=p0)
    FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0)) * coeff[2]

    print ('Fitted mean = ', coeff[1])
    print ('Fitted standard deviation = ', coeff[2])
    print ('Fitted FWHM = ', FWHM)

    return coeff, FWHM

def calculate_JJ(K):

    J_arg = K**2 / (1.0 + K**2) / 2.0
    JJ = j0(J_arg) - j1(J_arg)

    return JJ

#slices, zsep, Nslice, xlamds = load_slices('/home/alex/Desktop/pyGENT/genesis_run_150MW_tdp/mod_620.out')
#power_fft = power_spectrum(slices)
#omega = freq_domain_eV(zsep, Nslice, xlamds)

#plot omega vs power_fft

