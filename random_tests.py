# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

def extract_HERMES_wavelength(header):
    
    CRVAL1 = header['CRVAL1'] # / Co-ordinate value of axis 1                    
    CDELT1 = header['CDELT1'] #  / Co-ordinate increment along axis 1             
    CRPIX1 = header['CRPIX1'] #  / Reference pixel along axis 1                   
    
    print CRVAL1,CDELT1, CRPIX1
    
    #Creates an array of offset wavelength from the referece px/wavelength
    Lambda = CRVAL1 - (CRPIX1 - (np.arange(int(CRPIX1)*2)) -1)* CDELT1

    return Lambda

# <codecell>

CRVAL1 = 4714.9999999
CDELTA1 = .045177
CRPIX1 = 1.
data_len = 4096

# <codecell>

wl = np.arange(data_len)*CDELTA1+CRVAL1

# <codecell>

wl

# <codecell>

myfile = pf.open('21aug10036red.fits')

# <codecell>

extract_HERMES_wavelength(myfile[0].header)

# <codecell>

myfile[0].header.items()

# <codecell>

flux1 = myfile[0].data[5]
flux2 = myfile[0].data[20]

# <codecell>

plt.plot(flux1)
plt.plot(flux2)
    
plt.show()

    

# <codecell>

plt.plot(np.correlate(flux1, flux2, 'same'))
plt.show()

# <codecell>

filename = 'Giant01.obj'
# filename = 'red_Giant01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

# <codecell>

thisCam = thisStar.exposures.cameras[1]

# <codecell>

import numpy as np
import pickle
import pylab as plt
from scipy import interpolate, signal, optimize, constants
import pyfits as pf
import sys

CCReferenceSet=1
CCTHisSet = 2
corrHWidth = 10

lambda1, flux1 = thisCam.wavelengths[CCReferenceSet], thisCam.red_fluxes[CCReferenceSet]
# plt.plot(lambda1,flux1)
lambda2, flux2 = thisCam.wavelengths[CCTHisSet], thisCam.red_fluxes[CCTHisSet]
# plt.plot(lambda2,flux2)
# plt.show()

lambda1, flux1 = clean_flux(thisCam.wavelengths[CCReferenceSet], thisCam.red_fluxes[CCReferenceSet], thisCam)
# plt.plot(lambda1,flux1)
lambda2, flux2 = clean_flux(thisCam.wavelengths[CCTHisSet], thisCam.red_fluxes[CCTHisSet], thisCam)
# plt.plot(lambda2,flux2)
# plt.show()

CCCurve = signal.fftconvolve(flux1, flux2[::-1], mode='same')
CCCurve2 = signal.fftconvolve(flux1[-np.isnan(flux1)], flux2[-np.isnan(flux2)][::-1], mode='same')
print np.sum(-np.isnan(flux1)), len(flux1)
# corrMax = np.where(CCCurve==max(CCCurve))[0][0]

# p_guess = [corrMax,corrHWidth]
# x_mask = np.arange(corrMax-corrHWidth, corrMax+corrHWidth+1)
# p = fit_gaussian(p_guess, CCCurve[x_mask], np.arange(len(CCCurve))[x_mask])[0]

plt.plot(CCCurve)
plt.plot(CCCurve2)
# plt.plot(lambda2[x_mask],max(CCCurve)* gaussian(x_mask, p[0],p[1]))
plt.show()

# if np.modf(CCCurve.shape[0]/2.0)[0]>1e-5:
#     pixelShift = (p[0]-(CCCurve.shape[0]-1)/2.) #odd number of elements
# else:
#     pixelShift = (p[0]-(CCCurve.shape[0])/2.) #even number of elements

    
# # thisQ, thisdRV = QdRV(thisCam.wavelengths[i], thisCam.red_fluxes[i])
                
# mid_px = thisCam.wavelengths.shape[1]/2
# dWl = (thisCam.wavelengths[i,mid_px+1]-thisCam.wavelengths[i,mid_px]) / thisCam.wavelengths[i,mid_px]
# RV = dWl * pixelShift * constants.c 
# print 'RV',RV
                
# #                 SNR = np.median(thisCam.red_fluxes[i])/np.std(thisCam.red_fluxes[i])

# <codecell>

validDates = np.nansum(thisCam.red_fluxes,1).astype(bool)
print np.nansum(thisCam.wavelengths,1)

# <codecell>

print thisCam.RVs
# plt.plot(thisCam.RVs)
# plt.show()

# <codecell>

i=0
for x,y,label in zip(thisCam.wavelengths, thisCam.red_fluxes, thisCam.fileNames):
    plt.plot(x,y+i, label= label, c='k')
    i+=300
plt.title(thisStar.name)
# plt.legend(loc = 0)
plt.show()

# <codecell>

for i in a.sigmas[:]:
    plt.plot(i)
plt.show()

# <codecell>

#Fit gaussian in CCCurves
def gaussian(x, mu, sig, ):
    return np.exp(-np.power(x - mu, 2.) / 2 / np.power(sig, 2.))

def fit_gaussian(p, flux, x_range):
    a = optimize.leastsq(diff_gausian, p, args= [flux, x_range])
    return a

def diff_gausian(p, args):
    
    flux = args[0]
    x_range = args[1]
    diff = gaussian(x_range, p[0],p[1]) - flux/np.max(flux)
    return diff

def get_wavelength(wavelengths, pixel):
    intPx = int(pixel)
    fracPx = pixel - int(pixel)

    return (wavelengths[intPx+1] - wavelengths[intPx])*fracPx + wavelengths[intPx]

def extract_HERMES_wavelength(fileName):

	a = pf.open(fileName)

	CRVAL1 = a[0].header['CRVAL1'] # / Co-ordinate value of axis 1                    
	CDELT1 = a[0].header['CDELT1'] #  / Co-ordinate increment along axis 1             
	CRPIX1 = a[0].header['CRPIX1'] #  / Reference pixel along axis 1                   
	
	#Creates an array of offset wavelength from the referece px/wavelength
	Lambda = CRVAL1 - (CRPIX1 - (np.arange(int(CRPIX1)*2)) -1)* CDELT1

	return Lambda

# <codecell>

np.sum(thisCam.wavelengths,1)
np.sum(np.isnan(thisCam.wavelengths))

# <codecell>

mid_px = thisCam.wavelengths.shape[1]/2
dWl = (thisCam.wavelengths[0,mid_px+1]-thisCam.wavelengths[0,mid_px]) / thisCam.wavelengths[0,mid_px]
RV = dWl * 0.5 * 3e8
print 'RV',RV, mid_px, thisCam.wavelengths[0,mid_px+1], thisCam.wavelengths[0,mid_px]

# <codecell>

def clean_flux(wavelength, flux, thisCam, xDef = 1, medianRange = 0):
    '''Clean a 1D spectrum. 
    
    Parameters
    ----
    xDef : int or None, optional
        Coeficient to resample. Final array will be flux.shape[0]*xDef long. 
        
    medianRange : int, optional
        Number of pixels to median over. 0 will skip this step. Optional.

    '''
        
    #median outliers
    if medianRange>0:
        fluxMed = signal.medfilt(flux,medianRange)
        w = np.where(abs((flux-fluxMed)/np.maximum(fluxMed,50)) > 0.4)
        for ix in w[0]:
            flux[ix] = fluxMed[ix]
            
    print 'flux out has',np.sum(np.isnan(flux))

    if ((wavelength[-np.isnan(flux)].shape[0]>0) &  (flux[-np.isnan(flux)].shape[0]>0)):
        
        #flatten curve by fitting a 3rd order poly
        fFlux = optimize.curve_fit(cubic, wavelength[-np.isnan(flux)], flux[-np.isnan(flux)], p0 = [1,1,1,1])
        fittedCurve = cubic(wavelength, fFlux[0][0], fFlux[0][1], fFlux[0][2], fFlux[0][3])
        flux = flux/fittedCurve-1

        #apply tukey
        flux = flux * tukey(0.1, len(flux))

        #resample
        if (xDef>1):
            fFlux = interpolate.interp1d(wavelength, flux) 
            wavelength = np.linspace(min(wavelength), max(wavelength),len(wavelength)*xDef)
            flux = fFlux(wavelength)

    else: #if not enough data return NaNs
        if (xDef>1):
            wavelength = np.linspace(min(wavelength), max(wavelength),len(wavelength)*xDef)
            flux = np.ones(wavelength.shape[0])*np.nan
    
    return wavelength, flux

# <codecell>

def cubic(x,a,b,c,d):
    '''
    Cubic function
    '''
    return a*x**3+b*x**2+c*x+d

# <codecell>

def tukey(alpha, N):
    '''Creates a tukey function
    
    
    Parameters
    ----
    alpha : float
        Fraction of the pixels to fade in/out.
        i.e. alpha=0.1 will use 10% of the pixels to go from 0 to 1. 
        
    N : int
        Totla number of pixels in the array.
        
        
    Returns
    ------

    N-length array of floats from 0 to 1. 
    '''

    
    tukey = np.zeros(N)
    for i in range(int(alpha*(N-1)/2)):
        tukey[i] = 0.5*(1+np.cos(np.pi*(2*i/alpha/(N-1)-1)))
    for i in range(int(alpha*(N-1)/2),int((N-1)*(1-alpha/2))):
        tukey[i] = 1
    for i in range(int((N-1)*(1-alpha/2)),int((N-1))):
        tukey[i] = 0.5*(1+np.cos(np.pi*(2*i/alpha/(N-1)-2/alpha+1)))
    
    return tukey

# <codecell>

a

# <codecell>

thisStar.exposures.cameras[3].sigmas

# <codecell>


print thisCam.fileNames.shape

print thisCam.wavelengths.shape

# <codecell>

print thisCam.RVs
plt.plot(thisCam.RVs,'.')
plt.show()

# <codecell>

import pickle
import pylab as plt
import numpy as np

# <codecell>

cd ~/Documents/HERMES/reductions/HD1581_6.2/

# <codecell>

cd ~/Documents/HERMES/reductions/47Tuc_core_6.2/

# <codecell>

# filename = 'Giant01.obj'
# filename = 'Brght01.obj'
filename = 'red_Giant01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

# <codecell>

print np.all([np.nansum(thisCam.red_fluxes,1).astype(bool) for thisCam in thisStar.exposures.cameras],0)

# <codecell>

plt.plot(thisCam.red_fluxes[0])
plt.show()

# <codecell>

thisCam = thisStar.exposures.cameras[3]

# <codecell>

print thisCam.SNRs
print np.nansum(thisCam.red_fluxes,1)
print thisCam.fileNames

# <codecell>

thisStar.exposures.JDs.shape[0]

# <codecell>

thisStar.exposures.JDs.shape

# <codecell>

import scipy as sp

# <codecell>

from scipy import optimize

# <codecell>

thisStar.exposures.abs_baryVels

# <codecell>

pwd

# <codecell>

print RVs[0].shape

# <codecell>

cd '/Users/Carlos/Documents/HERMES/reductions/47Tuc_core_6.2'

# <codecell>

RVs = np.load('RVs.npy')
SNRs = np.load('SNRs.npy')

# <codecell>

for epoch in range(RVs.shape[1]):
    cam = 0
    R = RVs[:,epoch,cam]
    S = SNRs[:,epoch,cam]
    a = np.histogram(R)
    plt.bar(a[1][:-1],a[0], width = (a[1][-2]-a[1][-1])*0.7)
    plt.plot(R,S,'.', c='r')
    plt.show()

# <codecell>

a = np.histogram(R)
# plt.plot(a[1][:-1],a[0],'.')
plt.bar(a[1][:-1],a[0], width = (a[1][-2]-a[1][-1])*0.7)
plt.show()

# <codecell>

a = np.histogram(R)
plt.plot(R,S,'.', c='r')
# plt.bar(a[1][:-1],a[0], width = (a[1][-2]-a[1][-1])*0.7)
plt.show()

# <codecell>

R = RVs[:,15,0]
S = SNRs[:,15,0]

# <codecell>

a = np.histogram(R)
# plt.plot(a[1][:-1],a[0],'.')
plt.bar(a[1][:-1],a[0], width = (a[1][-2]-a[1][-1])*0.7)
plt.show()

# <codecell>

import os

# <codecell>

os.curdir

# <codecell>

pwd

# <codecell>

os.getcwd().split('/')[-1]

# <codecell>

plt.plot(thisCam.red_fluxes)
plt.show()

# <codecell>

i=6
np.nanmean(thisCam.red_fluxes[i])/np.std(thisCam.red_fluxes[i])
print np.sqrt(np.nanmean(thisCam.red_fluxes[i]))
print np.nansum(thisCam.red_fluxes[i])
print stats.nanmedian(thisCam.red_fluxes[i])/stats.nanstd(thisCam.red_fluxes[i])

# <codecell>

from scipy import stats

# <codecell>

print np.sqrt(stats.nanmedian(thisCam.red_fluxes[i]))

# <codecell>


