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

cd ~/Documents/HERMES/reductions/47Tuc_core_6.2/

# <codecell>

import pickle
import pylab as plt
import numpy as np

# <codecell>

filename = 'Brght24.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

# <codecell>

thisCam = thisStar.exposures.cameras[2]

# <codecell>

print np.nansum(thisCam.red_fluxes,1)
print np.nansum(thisCam.wavelengths,1)

# <codecell>

for i in a.sigmas[:]:
    plt.plot(i)
plt.show()

# <codecell>

i=0
for x,y,label in zip(thisCam.wavelengths, thisCam.red_fluxes, thisCam.fileNames):
    plt.plot(x,y+i, label= label, c='k')
    i+=300
plt.title(thisStar.name)
# plt.legend(loc = 0)
plt.show()

# <codecell>

N104-S2214.obj

