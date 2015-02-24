# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pickle
import create_obj as cr_obj
reload(cr_obj)
import RVTools as RVT
reload(RVT)
import pylab as plt
from scipy import signal
import numpy as np
from scipy import optimize, constants

# <codecell>

#opens a single star
filename = '/Users/Carlos/Documents/HERMES/reductions/HD1581_6.0/uncombined/HD1581.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

# <codecell>

thisCam = thisStar.exposures.cameras[0]
lambda1, flux1 = RVT.clean_flux(thisCam.wavelengths[0], thisCam.red_fluxes[0], thisCam)
lambda2, flux2 = RVT.clean_flux(thisCam.wavelengths[5], thisCam.red_fluxes[5], thisCam)

CCCurve = signal.convolve(flux1, flux2[::-1], mode='same')
corrMax = np.where(CCCurve==max(CCCurve))[0][0]
p_guess = [corrMax,10]
x_mask = np.arange(corrMax-2, corrMax+2+1)
p = fit_gaussian(p_guess, CCCurve[x_mask], np.arange(len(CCCurve))[x_mask])[0]

if np.modf(CCCurve.shape[0]/2.0)[0]>1e-5:
    pixelShift = (p[0]-(CCCurve.shape[0]-1)/2.) #odd number of elements
else:
    pixelShift = (p[0]-(CCCurve.shape[0])/2.) #even number of elements

# thisQ, thisdRV = QdRV(thisCam.wavelengths[i], thisCam.red_fluxes[i])

RV = np.array((thisCam.wavelengths[0,1]-thisCam.wavelengths[0,0])
              /thisCam.wavelengths[0,0]*constants.c*
              pixelShift)
print RV, pixelShift

# <codecell>

plt.plot(signal.convolve(flux1, flux2[::-1], mode='same'))
plt.plot(signal.fftconvolve(flux1, flux2[::-1], mode='same'))
plt.show()

# <codecell>

a = np.zeros(1000)
a[0]=1
b = np.zeros(1000)
b[10]=1

Fa=np.fft.fft(a)
Fb=np.fft.fft(b)

FaR = np.real(Fa)
FbR = np.real(Fb)

FaI = np.imag(Fa)
FbI = np.imag(Fb)


FaFb = Fa/Fb
FaFbR = np.real(FaFb)
FaFbI = np.imag(FaFb)

# <codecell>

# plt.plot(Fa, '.')
# plt.plot(Fb, '.')
# plt.plot(FaR,'.')
# plt.plot(FaI,'.')
# plt.plot(FbR,'.')
# plt.plot(FbI,'.')
# plt.plot(FaFb)
plt.plot(FaFbR/np.cos(np.arange(len(a))*.061))
plt.plot(np.cos(np.arange(len(a))*.061))
# plt.plot(np.arcsin(FaFbI),'.')
plt.show()

# <codecell>

plt.plot(np.angle(np.conjugate(np.fft.fft(flux1))*np.fft.fft(flux2)))
plt.show()

# <codecell>


plt.plot(flux1)
plt.plot(flux2)
plt.show()

# <codecell>

ftconv = np.conjugate(np.fft.fft(flux1))*np.fft.fft(flux2)
plt.plot(np.abs(ftconv))
plt.show()

# <codecell>

ww = np.where(np.abs(ftconv) > 15)
plt.plot(ww[0],np.angle(ftconv[ww[0]]),'.')
plt.show()

# <codecell>

test = np.convolve(ftconv,np.ones(4),mode='same')

# <codecell>

ww = np.where(np.abs(test) > 10)
plt.plot(ww[0],np.angle(test[ww[0]]),'.')
plt.show()

# <codecell>

test = np.convolve(ftconv,np.ones(4)/4.0,mode='same')
ww = np.where(np.abs(test) > 10)
plt.plot(ww[0],np.angle(test[ww[0]]),'.')
plt.show()

# <codecell>

x = ww[0][:665]
y = np.angle(test[ww[0]])[:665]

# <codecell>

a = np.polyfit(x, y, 1)
y2 = a[0]*x +a[1]
plt.plot(x,y, '.')
# plt.plot(x, y2)
plt.errorbar(x,y2, y-y2, y-y2)
plt.show()

# <codecell>

plt.plot(np.convolve(np.fft.fft(flux1),np.fft.fft(flux2),'same'))
plt.plot(np.fft.fft(flux1*flux2))
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


