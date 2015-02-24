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


