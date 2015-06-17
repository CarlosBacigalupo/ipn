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

np.nanmax(thisCam.red_fluxes, 1)

# <codecell>

data=np.load('npy/data.npy')

# <codecell>

data[:,2].astype(float)

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


# <codecell>


