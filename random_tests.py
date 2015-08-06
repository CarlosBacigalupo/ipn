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

import pylab as plt
import pyfits as pf

# <codecell>

a = pf.open('25aug10035.ms.fits')

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

import pickle
import toolbox

# <codecell>

cd /Users/Carlos/Documents/HERMES/reductions/6.5/rhoTuc/obj/

# <codecell>

thisCam = thisStar.exposures.cameras[3]

# <codecell>

thisCam.RVs

# <codecell>

import numpy as np
import pickle
import pylab as plt
from scipy import interpolate, signal, optimize, constants
import pyfits as pf
import sys
import RVTools as RVT
reload(RVT)
import time

# filename = 'HD1581.obj'
# filename = 'Brght01.obj'
# filename = 'red_Giant01.obj'
# filename = 'Giant01.obj'
filename = 'red_Giant12.obj'
# filename = 'Field01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

thisCam = thisStar.exposures.cameras[0]


# <codecell>

import numpy as np
import pickle
import pylab as plt
from scipy import interpolate, signal, optimize, constants
import pyfits as pf
import sys
import RVTools as RVT
reload(RVT)
import time

# filename = 'HD1581.obj'
# filename = 'Brght01.obj'
# filename = 'red_Giant01.obj'
# filename = 'Giant01.obj'
filename = 'Giant12.obj'
# filename = 'Field01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

thisCam = thisStar.exposures.cameras[0]

CCReferenceSet=0
CCTHisSet = 2
corrHWidth = 3
for CCTHisSet in range(15):
    # lambda1, flux1 = thisCam.wavelengths[CCReferenceSet], thisCam.red_fluxes[CCReferenceSet]
    # plt.plot(lambda1,flux1)
    # lambda2, flux2 = thisCam.wavelengths[CCTHisSet], thisCam.red_fluxes[CCTHisSet]
    # plt.plot(lambda2,flux2)
    # plt.show()

    lambda1, flux1 = RVT.clean_flux(thisCam.wavelengths[CCReferenceSet], thisCam.red_fluxes[CCReferenceSet], thisCam, medianRange = 5)
    plt.plot(thisCam.red_fluxes[CCTHisSet])
#     plt.plot(lambda1,flux1)
    lambda2, flux2 = RVT.clean_flux(thisCam.wavelengths[CCTHisSet], thisCam.red_fluxes[CCTHisSet], thisCam, medianRange = 5)
#     plt.plot(lambda2,flux2)
    plt.show()

    CCCurve = signal.fftconvolve(flux1, flux2[::-1], mode='same')
#     CCCurve2 = signal.fftconvolve(flux1[-np.isnan(flux1)], flux2[-np.isnan(flux2)][::-1], mode='same')
    # print np.sum(-np.isnan(flux1)), len(flux1)
    corrMax = np.where(CCCurve==max(CCCurve))[0][0]

    p_guess = [corrMax,corrHWidth]
    x_mask = np.arange(corrMax-corrHWidth, corrMax+corrHWidth+1)
    p = RVT.fit_gaussian(p_guess, CCCurve[x_mask], np.arange(len(CCCurve))[x_mask])[0]

#     plt.plot(lambda1,CCCurve/np.max(CCCurve))
    # plt.plot(CCCurve2)
    # plt.plot(lambda2[x_mask],max(CCCurve)* gaussian(x_mask, p[0],p[1]))
    plt.show()

    if np.modf(CCCurve.shape[0]/2.0)[0]>1e-5:
        pixelShift = (p[0]-(CCCurve.shape[0]-1)/2.) #odd number of elements
    else:
        pixelShift = (p[0]-(CCCurve.shape[0])/2.) #even number of elements


    # # thisQ, thisdRV = QdRV(thisCam.wavelengths[i], thisCam.red_fluxes[i])

    mid_px = thisCam.wavelengths.shape[1]/2
    dWl = (thisCam.wavelengths[CCReferenceSet,mid_px+1]-thisCam.wavelengths[CCReferenceSet,mid_px]) / thisCam.wavelengths[CCReferenceSet,mid_px]
    RV = dWl * pixelShift * constants.c 
    print CCTHisSet,'RV',RV,
    print

    # #                 SNR = np.median(thisCam.red_fluxes[i])/np.std(thisCam.red_fluxes[i])

# <codecell>


# <codecell>

import pickle
filename = 'HD1581.obj'
# filename = 'Brght01.obj'
# filename = 'red_Giant01.obj'
# filename = 'Giant01.obj'
# filename = 'Field01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

thisCam = thisStar.exposures.cameras[0]
CCReferenceSet=3
lambda1, flux1 = thisCam.wavelengths[CCReferenceSet], thisCam.red_fluxes[CCReferenceSet]
plt.plot(lambda1,flux1)

lambda2, flux2, fluxMed = clean_flux(thisCam.wavelengths[CCReferenceSet], thisCam.red_fluxes[CCReferenceSet], thisCam, medianRange = 5)
plt.plot(lambda2,flux2)
plt.plot(lambda2,fluxMed)
plt.show()

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
        fluxDiff = abs(flux-fluxMed)
        fluxDiffStd = np.std(fluxDiff)
        mask = fluxDiff> 2 * fluxDiffStd
        flux[mask] = fluxMed[mask]


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
        
    return wavelength, flux, fluxMed


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

for i in thisCam.red_fluxes:
    plt.plot(i)
plt.show()
# plt.plot(thisCam.RVs)
# plt.show()

# <codecell>

for flux,mj in zip(thisCam.red_fluxes,thisStar.exposures.JDs):
    plt.plot(flux+mj)
plt.show()
# plt.plot(thisCam.RVs)
# plt.show()

# <codecell>

np.sum(thisCam.red_fluxes, axis=1)

# <codecell>

print thisStar.exposures.JDs

# <codecell>

ls

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

cd ~/Documents/HERMES/reductions/47Tuc_core_6.2/

# <codecell>

print np.all([np.nansum(thisCam.red_fluxes,1).astype(bool) for thisCam in thisStar.exposures.cameras],0)

# <codecell>

plt.plot(thisCam.red_fluxes[0])
plt.show()

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

PyAstronomy

# <codecell>

import PyAstronomy

# <codecell>

cd ~/Documents/HERMES/reductions/HD1581_1arc_6.2/obj/

# <codecell>

import pickle
# filename = 'HD1581.obj'
# filename = 'Brght01.obj'
# filename = 'red_Giant01.obj'
# filename = 'Giant01.obj'
filename = 'Field01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)
thisCam = thisStar.exposures.cameras[3]

# <codecell>

import pylab as plt
plt.plot(thisCam.red_fluxes[0])
plt.show()

# <codecell>

from PyAstronomy import pyasl

def baryTest(thisStar):

    for i,jd in enumerate((thisStar.exposures.JDs+2400000.5)[:]):
        heli, bary = pyasl.baryvel(jd, deq=2000.0)
#         print "Earth's velocity at JD: ", jd
#         print "Heliocentric velocity [km/s]: ", heli
#         print "Barycentric velocity [km/s] : ", bary

        # Coordinates of Sirius
        ra  = 101.28715535
        dec = -16.71611587
        
        #thisStar coords
        ra  = np.rad2deg(thisStar.RA_dec)
        dec = np.rad2deg(thisStar.Dec_dec)
        print np.rad2deg(thisStar.RA_dec), np.rad2deg(thisStar.Dec_dec), thisStar.name, thisStar.exposures.abs_baryVels[i]

        
        vh, vb = pyasl.baryCorr(jd, ra, dec, deq=2000.0)
        print "Barycentric velocity of Earth toward",thisStar.name,'[m/s]', vb*1000
        print vb*1000-thisStar.exposures.abs_baryVels[i]
        print ''

# <codecell>

from PyAstronomy import pyasl

def baryTest2(baryVels, JDs):

    for i,jd in enumerate((JDs+2400000.5)[:]):
        heli, bary = pyasl.baryvel(jd, deq=2000.0)
#         print "Earth's velocity at JD: ", jd
#         print "Heliocentric velocity [km/s]: ", heli
#         print "Barycentric velocity [km/s] : ", bary

        # Coordinates of Sirius
        ra  = 101.28715535
        dec = -16.71611587
        
        #thisStar coords
        ra  = np.rad2deg(thisStar.RA_dec)
        dec = np.rad2deg(thisStar.Dec_dec)
        print np.rad2deg(thisStar.RA_dec), np.rad2deg(thisStar.Dec_dec), thisStar.name, thisStar.exposures.abs_baryVels[i]

        
        vh, vb = pyasl.baryCorr(jd, ra, dec, deq=2000.0)
        print "Barycentric velocity of Earth toward",thisStar.name,'[m/s]', vb*1000
        print vb*1000-thisStar.exposures.abs_baryVels[i]
        print ''

# <codecell>

import numpy as np
baryTest(thisStar)

# <codecell>

Right ascension	00h 20m 04.25995s
Declination	−64° 52′ 29.2549″

# <codecell>

print thisStar.RA_dec, thisStar.Dec_dec#, thisStar.RA_h, thisStar.RA_min , thisStar.RA_sec

# <codecell>

import toolbox
toolbox.dec2sex(thisStar.RA_dec)

# <codecell>

import numpy as np

# <codecell>

thisStar.RA_dec, toolbox.dec2sex(np.rad2deg(thisStar.RA_dec)/15)

# <codecell>

00 20 06.49 -64 52 06.6

# <codecell>

np.deg2rad(toolbox.sex2dec(0,20,06.49)*15)

# <codecell>

np.deg2rad(-toolbox.sex2dec(64,52,6.6))

# <codecell>

-toolbox.sex2dec(64,52,6.6)

# <codecell>

RVs = np.random.random(100)
stdRV= np.std(RVs)
medRV = 0.5
sigmaClip = 0.1
print RVs,stdRV,medRV
print RVs[(RVs>=medRV-sigmaClip*stdRV) & (RVs<=medRV+sigmaClip*stdRV)]

# <codecell>

import pylab as plt

# <codecell>

g = plt.gca()
g.xaxis.majorTic
plt.show()

# <codecell>

import matplotlib.pyplot as plt


# <codecell>

fig = plt.figure()
# plt.title(title)

ax = fig.add_subplot(111)
# ax.bar(hist[1][1:],hist[0], width = (hist[1][-2]-hist[1][-1]))
# ax.grid()
# ax.set_ylabel('Counts')
# ax.set_xlabel('RV [m/s]')
# #         ax.set_ylim(0,10)

# ax2 = ax.twinx()
# ax2.scatter(R,S, c='r', s=100)
ax.bar(0,1 , width = (1), color='k')

# <codecell>

plt.show()

# <codecell>

pwd

# <codecell>

cd 47Tuc_core_6.2/

# <codecell>

cd obj

# <codecell>

import pickle
# filename = 'HD1581.obj'
# filename = 'Brght01.obj'
# filename = 'red_Giant01.obj'
filename = 'red_Brght01.obj'
# filename = 'Giant01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)
thisCam = thisStar.exposures.cameras[3]

# <codecell>

print thisStar.exposures
thisCam.fileNames

# <codecell>

thisCam.red_fluxes

# <codecell>

float(np.sum(np.isnan(SNRs)))/(SNRs.shape[0]*SNRs.shape[1]*SNRs.shape[2])*100

# <codecell>

import pylab as plt
plt.plot(RVs[:,:,0])
plt.show()

# <codecell>

    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
#     sigmas=np.load('npy/sigmas.npy')
#     baryVels=np.load('npy/baryVels.npy')
#     JDs=np.load('npy/JDs.npy')
    SNRs = np.load('npy/SNRs.npy')

# <codecell>

pwd

# <codecell>


filehandler = open('obj/red_N104-S1084.obj', 'r')
thisStar = pickle.load(filehandler)

# <codecell>

thisCam = thisStar.exposures.cameras[0]

# <headingcell level=1>

# Bary tests

# <codecell>

import toolbox as tb
from PyAstronomy import pyasl


# <codecell>

MJD = 57131.84792 
RA = tb.deg2rad(tb.sex2dec(0,24,05.67)*15) #47tuc in rad
Dec = tb.deg2rad(-tb.sex2dec(72,4,52.6)) #47Tuc in rad

# <codecell>

RA, Dec

# <codecell>

MJDs = []
RVs = []
RV2s = []
RV3s = []
for i in np.arange(MJD-200, MJD+200):
    vh, vb = tb.baryvel(i+2400000+0.5) 
    vh2, vb2 = pyasl.baryvel(i+2400000+0.5, deq = 0.0)
    __, RV3 = pyasl.baryCorr(i+2400000+0.5,tb.rad2deg(RA), tb.rad2deg(Dec))
    
    RV = (vb[0]*np.cos(Dec)*np.cos(RA) + vb[1]*np.cos(Dec)*np.sin(RA) + vb[2]*np.sin(Dec))*1000
    RV2 = (vb2[0]*np.cos(Dec)*np.cos(RA) + vb2[1]*np.cos(Dec)*np.sin(RA) + vb2[2]*np.sin(Dec))*1000
    MJDs.append(i)
    RVs.append(RV)
    RV2s.append(RV2)
    RV3s.append(RV3*1000)
RVs = np.array(RVs)
RV2s = np.array(RV2s)
RV3s = np.array(RV3s)

# <codecell>

plt.plot(MJDs, RVs)
plt.plot(MJDs, RV2s)
plt.plot(MJDs, RV3s)
plt.show()

plt.plot(MJDs, RVs-RV2s, label = '1-2')
plt.plot(MJDs, RV2s-RV3s, label = '2-3')
plt.legend(loc=0)
plt.show()


# <headingcell level=1>

# plot app

# <codecell>

import numpy as np
import pylab as plt
a = np.random.rand(10)
b = np.random.rand(10)
c = np.random.rand(10)
plt.scatter(a,b)
plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('custom picker for line data')
line, = ax1.plot(a, b, 'o', picker=5)
fig.canvas.mpl_connect('pick_event', onpick2)


def onpick2():
    a=2
    print a

# <codecell>

def aaa():
    xxx = 'sdd'
    print 'asdasd'
    return xxx

# <codecell>

def test():
    print 'adasd'

# <codecell>


# <codecell>


# <codecell>


# <codecell>

import glob
import numpy as np
import os

os.chdir('/Users/Carlos/Documents/HERMES/reductions/')
a = glob.glob('*')
for i in a:
    if i!='HD1581_6.0':
        print i,
        b = np.load(i+'/npy/data.npy')
        print b.shape[0]

# <codecell>

data

# <codecell>

cd HD285507_1arc_6.2/

# <codecell>

cd HERMES/reductions/HD285507_1arc_6.2/

# <codecell>

import glob
import numpy as np
import os
data=np.load('npy/data.npy')
RVs=np.load('npy/RVs.npy')
SNRs=np.load('npy/SNRs.npy')

# <codecell>

RVs.shape
W = np.zeros(np.hstack((RVs.shape, RVs.shape[0])))
for thisStarIdx in range(RVs.shape[0]):
    W1 = np.ones(RVs.shape)/(RVs.shape[0]-1)
    W1[thisStarIdx,:,:]=0
    W[:,:,:,thisStarIdx]=W1
    

# <codecell>

data[thisStarIdx,2].astype(float).astype(int)

# <codecell>

import RVTools as RVT
reload(RVT)
import pylab as plt
import pandas as pd

# <codecell>

data=np.load('npy/data.npy')
RVs=np.load('npy/RVs.npy')
SNRs=np.load('npy/SNRs.npy')

# <codecell>

# SNRs[:,:,0][SNRs[:,:,0]<1]
SNRs[np.isnan(SNRs)]=0
SNRs+=1e-17
create_allW(data,SNRs)

# <codecell>

import numpy as np
import pylab as plt

# <codecell>

cd /Users/Carlos/Documents/HERMES/reductions/6.2/HD285507_1arc_6.2/

# <codecell>

allW = np.load('npy/allW_DM.npy')
allW[0,:,0][0]

# <codecell>


# deltay = np.linspace(0, 4000)
# deltay = np.linspace(-2000, 2000)
# deltay = np.linspace(-4000, 0)
# SNRs = np.ones(50)*30
# SNRs = np.linspace(10, 100)
# SNRs = np.linspace(100, 10)
# W = calibrator_weights2(deltay,SNRs)

data=np.load('npy/data.npy')
# RVs=np.load('npy/RVs.npy')
SNRs=np.load('npy/SNRs.npy')
allW = np.load('npy/allW_PM.npy')
idx = np.where(data[:,0]=='Giant01')[0]
for cam in range(4):
    W = allW[:,cam,idx]

    thisSNRs = SNRs[:,0,cam]

#     plt.plot(deltay/np.max(np.abs(deltay)), label = 'deltay')
    plt.plot(thisSNRs, label= 'SNR')
    plt.plot(W*np.nanmax(thisSNRs), label = 'W')
    plt.legend(loc=0)
    # title = 'PM - deltay '+str(np.min(deltay))+','+str(np.max(deltay))+' - SNR '+str(np.min(SNRs))+','+str(np.max(SNRs))
#     title = 'DM - deltay '+str(np.min(deltay))+','+str(np.max(deltay))+' - SNR '+str(np.min(SNRs))+','+str(np.max(SNRs))
    title = 'PM - cam '+str(cam)+ ' ,' + str(data[idx])
    plt.title(title)
    plt.grid(True)
    plt.savefig(('PM_'+str(cam)))
    plt.show()

# <codecell>

import numpy as np

# <codecell>

def create_allW(data = [], SNRs = []):

    if ((data!=[]) and (SNRs!=[])):

        #load function that translates pivot# to y-pixel  p2y(pivot)=y-pixel of pivot
        p2y = RVT.pivot_to_y('/Users/Carlos/Documents/HERMES/reductions/6.2//rhoTuc_6.2/0_20aug/1/20aug10042tlm.fits') 

        #gets the y position of for the data array
        datay = p2y[data[:,2].astype(float).astype(int)]

        #Creates empty array for relative weights
        #allW[Weights, camera, staridx of the star to be corrected]
        allW = np.zeros((data.shape[0],4,data.shape[0]))

        for thisStarIdx in range(data.shape[0]):

            #converts datay into deltay
            deltay = datay-datay[thisStarIdx]
            for cam in range(2):

                thisSNRs = SNRs[:,0,cam].copy()
                thisSNRs[np.isnan(thisSNRs)]=1  #sets NaNs into SNR=1 

                W = calibrator_weights(deltay,thisSNRs)
                allW[:,cam,thisStarIdx] = W

                order = np.argsort(deltay)
                plt.plot(deltay[order], label = 'deltay')
                plt.plot(thisSNRs[order], label= 'SNR')
                plt.plot((W*np.max(deltay))[order], label = 'W')
                plt.legend(loc=0)
                plt.show()

        # a= pd.DataFrame(deltay)
        # # a.columns = labels
        # print a.head(n=40)
        # print W, np.sum(W)
    else:
        print 'Create allW: Input arrays missing'
        allW =[]

    return allW

# <codecell>

def create_RVCorr(RVs, allW, RVClip = 1e17):
    RVCorr = np.zeros(RVs.shape)
    RVs[np.abs(RVs)>RVClip]=0
    for thisStarIdx in range(data.shape[0]):
        for epoch in range(RVs.shape[1]):
            for cam in range(4):
                RVCorr[thisStarIdx,epoch,cam] = np.nansum(allW[:,cam,thisStarIdx]*RVs[:,epoch,cam])
    return RVCorr

# <codecell>

# for i in range(40):
i=37
plt.plot(RVs[i,:,0], label = 'RV')
plt.plot(RV_corr[i,:,0], label = 'Correction')
plt.plot(RVs[i,:,0]-RV_corr[i,:,0], label = 'Result', marker = 'o')
plt.legend(loc=0)
plt.show()

# <codecell>

np.where(data[:,0]=='Giant01')

# <codecell>

for epoch in range(RVs.shape[1]):
    print np.sum(RVs[:,epoch,1])/RVs.shape[1]

# <codecell>

def create_corrRVs(RVs,W):
#Creates corrRVs with RV corrections for each RV. 
#has the same shape than RVs and W
#RVs-corrRVs = trueRVs (values without systematics)

    corrRVs = np.ones(RVs.shape)*np.nan
    
    #1 - loop retarded method. should be array operation.
    
    #check shape
    if ((RVs.shape==W.shape[:3]) and (len(W.shape)==4) and (RVs.shape[0]==W.shape[3])):
        for thisStaridx in range(RVs.shape[0]):
            for epoch in range(RVs.shape[1]):
                for cam in range(RVs.shape[2]):
                    corrRVs[thisStaridx,epoch,cam] = np.sum(RVs[:,epoch,cam]*W[:,epoch,cam,thisStaridx])
                
    else:
        print 'Bad array shape.'
        print 'RVs=', RVs.shape
        print 'W=', W.shape
        
    return corrRVs

# <headingcell level=3>

# Solar Spectrum

# <codecell>

import pyfits as pf
import pylab as plt
import RVTools as RVT

fileList = glob.glob('cam1/*.fits')
b=[]
wl = RVT.extract_HERMES_wavelength(fileList[0])
#build fibre filter
a = pf.open(fileList[0])
filt = np.logical_or((a['FIBRES'].data['TYPE']=='P'),(a['FIBRES'].data['TYPE']=='S'))
filt[175]=False

plt.plot(wl,np.sum(a[0].data[filt], axis=0))
plt.show()



# for fits in fileList[:]:
#     print 'Reading',fits
#     a = pf.getdata(fits)
#     if b==[]:
#         b =a
#     else:
#         b+=a


# <headingcell level=3>

# SNR 3d plots

# <codecell>

SNRs=np.load('npy/SNRs.npy')
Data=np.load('npy/Data.npy')

labels = Data[:,0]

# <codecell>

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
X = np.arange(SNRs.shape[1])
Y = np.arange(SNRs.shape[0])
X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
Z = SNRs[:,:,0]
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0,  vmin=0, vmax=100, antialiased=True)
ax.set_xlabel('Epoch')
ax.set_ylabel('Star')
ax.set_zlabel('SNR')
# ax.set_zlim(-1.01, 1.01)
ax.set_yticks(np.arange(0,SNRs.shape[0],5))
# ax.set_yticklabels(labels)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


# <headingcell level=3>

# Comments array

# <codecell>

def comment(star, epoch, cam, comment):
    comments = []
    try:
        comments = np.load('npy/comments.npy')
    except:
        pass
    
    if comments==[]:
        comments = np.zeros((1,),dtype=('i4,i4,i4,a10'))
        comments[:] = [(star, epoch, cam, comment)]
    else:
        x = np.zeros((1,),dtype=('i4,i4,i4,a10'))
        x[:] = [(star, epoch, cam, comment)]
        print x,comments
        comments = np.append(comments,x)
    
    np.save('npy/comments.npy',comments)
        

# <codecell>

comment(0,0,0,'test')

# <codecell>

c = np.load('npy/comments.npy')
d = np.load('npy/data.npy')

# <codecell>

d[1]

# <codecell>

filename = 'obj/Field03.obj'
# filename = 'red_Giant01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

# <codecell>

from scipy import stats

# <codecell>

thisCam = thisStar.exposures.cameras[0]

# <codecell>

stats.nanmedian(thisCam.red_fluxes[13])

# <codecell>

c

# <codecell>

[np.asarray(a), np.asarray(a)]

# <codecell>

x = np.zeros((2,),dtype=('i4,i4,i4,a10'))
x[:] = [(1,2,3,'Hello'),(2,3,4,"World")]

# <codecell>

np.append(x,x)

# <codecell>

np.vstack((x,(1,2,3,'Hello')))

# <headingcell level=3>

# Check RVCorr

# <codecell>

data=np.load('npy/data.npy')
RVs=np.load('npy/RVs.npy')
sigmas=np.load('npy/sigmas.npy')
baryVels=np.load('npy/baryVels.npy')
JDs=np.load('npy/JDs.npy')    
# RVCorr_PM=np.load('npy/RVCorr_PM.npy')
# # RVCorr_DM=np.load('npy/RVCorr_DM.npy')
# cRVs_PM=np.load('npy/cRVs_PM.npy')
# cRVs_DM=np.load('npy/cRVs_DM.npy')
# cRVs_PMDM=np.load('npy/cRVs_PMDM.npy')

# <codecell>

idx = np.where(data[:,0]=='Giant01')[0][0]

# <codecell>

starIdx = idx
cam = 0

RVs[RVs>5000]=np.nan
RVs[RVs<-5000]=np.nan

# <codecell>

plt.plot(RVs[starIdx,:,cam])
# plt.plot(RVCorr_DM[starIdx,:,cam])
plt.plot(RVCorr_PM[starIdx,:,cam])
# plt.plot(RVs[starIdx,:,cam]-RVCorr_DM[starIdx,:,cam])
plt.plot(RVs[starIdx,:,cam]-RVCorr_PM[starIdx,:,cam])
plt.show()

# <codecell>

reload(RVT)

# <codecell>

allW_PM = RVT.create_allW(data, SNRs, starSet = [], RVCorrMethod = 'PM', refEpoch = 0) 
# RVCorr_PM = RVT.create_RVCorr_PM(RVs, allW_PM, RVClip = 2000, starSet = [])

# <codecell>

plt.plot(allW_PM[:,0,starIdx])
plt.plot(SNRs[:,0,starIdx])
plt.plot(W)
plt.show()

# <codecell>

data[starIdx]

# <codecell>

p2y = RVT.pivot_to_y('/Users/Carlos/Documents/HERMES/reductions/6.2/rhoTuc_6.2/0_20aug/1/20aug10042tlm.fits') 
datay = p2y[data[:,2].astype(float).astype(int)]
deltay = datay-datay[starIdx]

thisSigma = 1./SNRs[:,0,0].copy()
thisSigma[np.isnan(thisSigma)]=1e+17  #sets NaNs into SNR=1e-17
W = RVT.calibrator_weights(deltay,thisSigma)

# <codecell>

W

# <codecell>

import psycopg2 as mdb
con = mdb.connect("dbname=hermes_master user=Carlos")
cur = con.cursor()
cur.execute("CREATE TABLE fields(id int)")

# <codecell>

con.rollback()

# <codecell>

con.commit()

# <codecell>

con.close()

# <codecell>

import psycopg2 as mdb
con=mdb.connect("host=/tmp/ dbname=hermes_master user=Carlos");

# <codecell>

con=mdb.connect("dbname=hermes_master user=Carlos");

# <codecell>


con=mdb.connect("host=/usr/local/var dbname=hermes_master user=Carlos");

# <codecell>

cur = con.cursor()

# <codecell>

cur.execute("SELECT spec_path,name from fields where ymd=140825 and ccd='ccd_1' and obstype='BIAS'")

# <codecell>

objs=cur.fetchall()

# <codecell>

from pyraf import iraf

# <codecell>

iraf.noao(_doprint=0,Stdout="/dev/null")
iraf.imred(_doprint=0,Stdout="/dev/null")
iraf.ccdred(_doprint=0,Stdout="/dev/null")

# <codecell>

iraf.ccdproc(images='tmp/flats/25aug10034.fits', ccdtype='', fixpix='no', oversca='no', trim='no', zerocor='yes', darkcor='no', flatcor='no', zero='tmp/masterbias',Stdout="/dev/null")

# <codecell>

pwd

# <codecell>

cd ~/Documents/workspace/GAP/IrafReduction/140825/ccd11/

# <codecell>

import pyfits as pf

# <codecell>

pf.open('tmp/masterbias.fits')

# <codecell>

import cosmics

# <codecell>

import numpy as np
import pylab as plt

# <codecell>

a = np.arange(5)
b = np.array([1,5,3,2,5])
c = np.arange(0.5,4.5)

# <codecell>

d = np.interp(c,a,b)

# <codecell>

b

# <codecell>

plt.plot(a,b,marker='+')
plt.scatter(c,d, marker = '+', s=200, c='r')
plt.show()

# <codecell>

import os
import create_obj as cr_obj
import pickle
import glob
import pyfits as pf
import numpy as np
import sys
import toolbox
import importlib

# <codecell>

booHD1581 = False
IRAFFiles = '/Users/Carlos/Documents/workspace/GAP/IrafReduction/results/'   #folder to IRAF reduced files
dataset = 'HD285507'

# <codecell>

os.mkdir('cam1')
os.mkdir('cam2')
os.mkdir('cam3')
os.mkdir('cam4')
os.mkdir('obj')

# <codecell>

thisDataset = importlib.import_module('data_sets.'+dataset)

    

# <codecell>

months = np.array(['', 'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
d = np.array([s[4:] for s in thisDataset.date_list])
m = months[np.array([s[2:4] for s in thisDataset.date_list]).astype(int)]
filename_prfx = np.core.defchararray.add(d, m)

# <codecell>

for i,folder in enumerate(thisDataset.date_list):
    for files in thisDataset.ix_array[i][2:]:
        for cam in range(1,5):
            strCopy = 'cp ' + IRAFFiles + folder + '/norm/' + filename_prfx[i] + str(cam) + "%04d" % (files,) + '.ms.fits ' 
            strCopy += 'cam'+ str(cam) + '/' + filename_prfx[i] + str(cam) + "%04d" % (files,) + '.fits ' 
            print strCopy
            try:
                os.system(strCopy)
            except:
                print 'no copy'

# <codecell>

thisDataset.ix_array[1][2:]

# <codecell>


# <codecell>

filename_prfx

# <codecell>

thisDataset.ix_array

# <headingcell level=3>

# CC arcs

# <codecell>

cd ~/Documents/HERMES/reductions/6.5/HD285507/

# <codecell>

import glob
import os
import importlib
import numpy as np

# <codecell>

#Copy are files
files = glob.glob('*')
thisDataset = importlib.import_module('data_sets.HD1581')

# for folderList in files:
#     try:
#     if int(folderList[:1]) in range(20):

#compose file prefixes from date_list
months = np.array(['', 'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
d = np.array([s[4:] for s in thisDataset.date_list])
m = months[np.array([s[2:4] for s in thisDataset.date_list]).astype(int)]
filename_prfx = np.core.defchararray.add(d, m)



for i,folder in enumerate(thisDataset.date_list):
    thisFile = "%04d" % thisDataset.ix_array[i][1]
    for cam in range(1,5):
        strCopy = 'cp ' + str(i) + '_' + filename_prfx[i] + '/' + str(cam) + '/' + filename_prfx[i] + str(cam) + thisFile + 'red.fits ' 
#         strCopy = str(i) + '_' + filename_prfx[i] + '/' + str(cam) + '/' + filename_prfx[i] + str(cam) + thisFile + 'red.fits ' 
#         strCopy += 'arc_cam'+ str(cam) + '/' + filename_prfx[i] + str(cam) + thisFile + 'red.fits ' 
        strCopy += '.' 
        print strCopy
        
        os.system(strCopy)

            
#     except:
#         print 'error'

# <codecell>

import pyfits as pf
import pylab as plt
import numpy as np
import RVTools as RVT
from scipy import signal, optimize, constants
import os
import glob
reload(RVT)

os.chdir('/Users/Carlos/Documents/HERMES/reductions/6.5/HD1581/')

corrHWidth = 5
xDef = 1
fibre = 30

arcRVs = np.ones((400,5,4))*np.nan

for cam in range(4):
    for fibre in [175]: range(400):
        print fibre
        files = glob.glob('arc_cam'+str(cam+1)+'/*')
        for i,thisFile in enumerate(files):
            if i==0:
                fits = pf.open(thisFile)
                refWL = RVT.extract_HERMES_wavelength(thisFile)
                refData = fits[0].data
                fits.close()

        #         print refWL.shape
        #         print refData.shape
        #         plt.plot(refWL, refData[101])
        #         plt.show()


            fits = pf.open(thisFile)
            thisWL = RVT.extract_HERMES_wavelength(thisFile)
            thisData = fits[0].data
            fits.close()

            lambda1, flux1 = RVT.clean_flux(refWL, refData[fibre], flatten = False)
    #         plt.plot(lambda1,flux1)
            lambda2, flux2 = RVT.clean_flux(thisWL, thisData[fibre], flatten = False)
    #         plt.plot(lambda2,flux2)
    #         plt.show()

            try:
                CCCurve = signal.fftconvolve(flux1[-np.isnan(flux1)], flux2[-np.isnan(flux2)][::-1], mode='same')
                corrMax = np.where(CCCurve==max(CCCurve))[0][0]
                p_guess = [corrMax,corrHWidth]
                x_mask = np.arange(corrMax-corrHWidth, corrMax+corrHWidth+1)
                if max(x_mask)<len(CCCurve):
                    p = fit_gaussian(p_guess, CCCurve[x_mask], np.arange(len(CCCurve))[x_mask])[0]
                    if np.modf(CCCurve.shape[0]/2.0)[0]>1e-5:
                        pixelShift = (p[0]-(CCCurve.shape[0]-1)/2.) #odd number of elements
                    else:
                        pixelShift = (p[0]-(CCCurve.shape[0])/2.) #even number of elements


                    mid_px = refData.shape[1]/2
                    dWl = (refWL[mid_px+1]-refWL[mid_px]) / refWL[mid_px]/xDef
                    RV = dWl * pixelShift * constants.c 
    #                 print 'RV',fibre,i,RV
                    arcRVs[fibre,i,cam] = RV
            except:
                pass

# <codecell>

cam =3
filename = 'HD1581arc_IR'

files = glob.glob('arc_cam'+str(cam+1)+'/*')
for i,thisFile in enumerate(files):
    fits = pf.open(thisFile)
    refWL = RVT.extract_HERMES_wavelength(thisFile)
    refData = fits[0].data
    fits.close()
    
    file_object = open(filename+'_e'+str(i)+'.txt', 'w')
    for wl,fl in zip(refWL,refData[175]):
        file_object.write(str(wl)+' '+str(fl)+'\n')
    file_object.close()

# <codecell>

import pickle
filename = '../obj/HD1581.obj'
# filename = 'Brght01.obj'
# filename = 'red_Giant01.obj'
# filename = 'Giant01.obj'
# filename = 'Field01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)
thisCam =  thisStar.exposures.cameras[0]

thisStar.exposures.pivots

# <codecell>

RVs = np.nanmean(arcRVs,axis=0)

# <codecell>

X = JDs[np.array([0,1,4,7,12])]

# <codecell>

JDs = np.load('../npy/JDs.npy')
plt.scatter(JDs,arcRVs[175])
plt.show()

# <codecell>

np.save('npy/arcRVs',arcRVs)

# <codecell>

print arcRVs.shape
arcRVs2 = np.ones((400,15,4))*np.nan
arcRVs2[:,0,:] = arcRVs[:,0,:]
arcRVs2[:,1,:] = arcRVs[:,1,:]
arcRVs2[:,2,:] = arcRVs[:,1,:]
arcRVs2[:,3,:] = arcRVs[:,1,:]
arcRVs2[:,4,:] = arcRVs[:,2,:]
arcRVs2[:,5,:] = arcRVs[:,2,:]
arcRVs2[:,6,:] = arcRVs[:,2,:]
arcRVs2[:,7,:] = arcRVs[:,3,:]
arcRVs2[:,8,:] = arcRVs[:,3,:]
arcRVs2[:,9,:] = arcRVs[:,3,:]
arcRVs2[:,10,:] = arcRVs[:,3,:]
arcRVs2[:,11,:] = arcRVs[:,3,:]
arcRVs2[:,12,:] = arcRVs[:,4,:]
arcRVs2[:,13,:] = arcRVs[:,4,:]
arcRVs2[:,14,:] = arcRVs[:,4,:]
arcRVs = arcRVs2
print arcRVs.shape

# <codecell>

arcRVs.shape

# <headingcell level=3>

# fibre ID table

# <codecell>

import pandas as pd

# <codecell>

np.tile(np.arange(10,0,-1),40)+np.repeat(np.arange(0,40)*10,10)

# <codecell>

labels = ['2dfID', 'idxData']


rev_num = np.tile(np.arange(10,0,-1),40)+np.repeat(np.arange(0,40)*10,10)

data = np.zeros((400,2))
# data[:] = ''
data[:,0] = np.arange(1,401)
data[:,1] = (rev_num-1)
# print data
# data[range(49,399,50),5] = 'Guiding fibre'
a = pd.DataFrame(data)
a.columns = labels

pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 500)

# <codecell>

a.ix[np.hstack((range(60),range(350,400)))]
a.ix[:20]

# <codecell>

file_object = open('a.txt', 'w')
file_object.write(a.to_latex(index=False))
file_object.close()

# <codecell>

a.to_latex(index=False)

# <codecell>

# ('NAME', 'S80'),
# ('RA', '>f8'), 
# ('DEC', '>f8'), 
# ('X', '>i4'), 
# ('Y', '>i4'), 
# ('XERR', '>i2'), 
# ('YERR', '>i2'), 
# ('THETA', '>f8'), 
# ('TYPE', 'S1'), 
# ('PIVOT', '>i2'), 
# ('MAGNITUDE', '>f8'), 
# ('PID', '>i4'), 
# ('COMMENT', 'S80'), 
# ('RETRACTOR', 'S10'), 
# ('WLEN', '>f8'), 
# ('PMRA', '>f8'), 
# ('PMDEC', '>f8')]

# <codecell>

('Giant01', 
 1.0778280795690982, 
 0.26764721042069262, 
 -15, 
 3, 
 15, 
 1,
 3.4963547846475636, 
 'P', 
 223, 
 7.6699999999999999, 0, 'Kmag', '23', 0.0, 0.0, 0.0)

# <codecell>

cd '/Users/Carlos/Documents/HERMES/reductions/6.5/HD1581/'

# <codecell>

# arcRVs = np.load('npy/arcRVs.npy')
JDs = np.load('npy/JDs.npy')


# <codecell>

JDs

# <codecell>

arcRVs2 = np.ones((400,16,4))*np.nan
arcRVs2[:,0,:] = arcRVs[:,0,:]
arcRVs2[:,1,:] = arcRVs[:,0,:]
arcRVs2[:,2,:] = arcRVs[:,0,:]
arcRVs2[:,3,:] = arcRVs[:,0,:]
arcRVs2[:,4,:] = arcRVs[:,1,:]
arcRVs2[:,5,:] = arcRVs[:,1,:]
arcRVs2[:,6,:] = arcRVs[:,1,:]
arcRVs2[:,7,:] = arcRVs[:,2,:]
arcRVs2[:,8,:] = arcRVs[:,2,:]
arcRVs2[:,9,:] = arcRVs[:,2,:]
arcRVs2[:,10,:] = arcRVs[:,3,:]
arcRVs2[:,11,:] = arcRVs[:,3,:]
arcRVs2[:,12,:] = arcRVs[:,3,:]
arcRVs2[:,13,:] = arcRVs[:,4,:]
arcRVs2[:,14,:] = arcRVs[:,4,:]
arcRVs2[:,15,:] = arcRVs[:,4,:]
arcRVs = arcRVs2
np.save('npy/arcRVs',arcRVs)

# <codecell>

bary = np.load('npy/baryVels.npy')

# <codecell>


# <codecell>

bary.shape

# <codecell>


# <codecell>

bary

# <codecell>

bary[:-1]-bary[1:]

# <codecell>

pwd

# <codecell>

a = np.array([4860,4865])
np.save('npy/cam1Filter.npy',a)
a = np.array([5751,5756])
np.save('npy/cam2Filter.npy',a)
a = np.array([6560,6565])
np.save('npy/cam3Filter.npy',a)
a = np.array([7710,7718])
np.save('npy/cam4Filter.npy',a)

# <codecell>

1.400E7
648.36
818.28
570.57
142.85
150.76
150.88
80.39
84.36
99.99
105.94
124.30
122.21
122.19
121.61

# <codecell>

hbetaRVs = (np.loadtxt('out3_hbeta.txt', delimiter = ' ', dtype='str')[:,26]).astype(float)

# <codecell>

arcRVs = np.loadtxt('arc_hbeta.txt', delimiter = ' ', dtype='str', usecols = [40]).astype(float)

# <codecell>

bary = np.load('npy/baryVels.npy')
JDs = np.load('npy/JDs.npy')

# <codecell>

JDs

# <codecell>

[0,1,4,7,12]

# <codecell>

plt.scatter(JDs,hbetaRVs-bary,color = 'k', s=100, marker='*', label = 'Stars')
plt.scatter(JDs[np.array([0,1,4,7,12])],arcRVs,  marker = '+' , label = 'ARC RV', color = 'm', s=500)
plt.show()

# <codecell>

arcRVs[0] = 1.4e-7

# <codecell>

bary

# <codecell>


