# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=3>

# 2dfdr, iraf, pyhermes compare

# <codecell>

import numpy as np
import pylab as plt
import pyfits as pf
from scipy.stats import nanmedian
import RVTools as RVT
reload(RVT)

# <codecell>

cd /Users/Carlos/Documents/HERMES/reductions

# <headingcell level=6>

# HD285507

# <codecell>

file2_1 = pf.open('6.5/HD285507/cam1/20aug10039red.fits')
file2_2 = pf.open('6.5/HD285507/cam1/20aug10040red.fits')
file2_3 = pf.open('6.5/HD285507/cam1/20aug10041red.fits')
filei_1 = pf.open('iraf/HD285507/cam1/20aug10039.fits')
filei_2 = pf.open('iraf/HD285507/cam1/20aug10040.fits')
filei_3 = pf.open('iraf/HD285507/cam1/20aug10041.fits')
# filep = pf.open('pyhermes/HD285507/combined/cam1/20aug10039comb.fits')
filep = pf.open('pyhermes/initial_RV_bulk/140820HD285507_p1/1408200037_FIB228_1.fits')

# <codecell>

RV = filep[0].header['v_bary']
RVShift = RV*4825/300000.

# <codecell>

file2_wl = RVT.extract_HERMES_wavelength('6.5/HD285507/cam1/20aug10039red.fits') + RVShift

# <codecell>

filei_wl = RVT.extract_iraf_wavelength(filei_1[0].header, 224)

# <codecell>

filep_wl = RVT.extract_pyhermes_wavelength('pyhermes/initial_RV_bulk/140820HD285507_p1/1408200037_FIB228_1.fits')

# <codecell>

file2_data = np.zeros((3,file2_1[0].data.shape[0],file2_1[0].data.shape[1]))
file2_data[0,:,:] = file2_1[0].data
file2_data[1,:,:] = file2_2[0].data
file2_data[2,:,:] = file2_3[0].data
file2_data = np.sum(file2_data, axis=0)

# <codecell>

filei_data = np.zeros((3,filei_1[0].data.shape[0],filei_1[0].data.shape[1]))
filei_data[0,:,:] = filei_1[0].data
filei_data[1,:,:] = filei_2[0].data
filei_data[2,:,:] = filei_3[0].data
filei_data = np.sum(filei_data, axis=0)

# <codecell>

filep_data = filep[0].data

# <codecell>

plt.plot(file2_wl, file2_data[227]/np.median(file2_data[227]), label = '2dfdr')
plt.plot(filei_wl, filei_data[223]/np.median(filei_data[223]), label = 'iraf')
plt.plot(filep_wl, filep_data/nanmedian(filep_data), label = 'pyhermes')
plt.legend(loc=0)
plt.show()

# <headingcell level=6>

# HD1581

# <codecell>

file2 = pf.open('6.5/HD1581/cam1/20aug10053red.fits')
filei = pf.open('iraf/HD1581/cam1/20aug10053.fits')
filep = pf.open('pyhermes/initial_RV_bulk/140820rhoTucNew_p0/1408200042_FIB176_1.fits')
filem = np.loadtxt('myherpy/HD1581_0.txt')

# <codecell>

RV = filep[0].header['v_bary']
RVShift = RV*4825/300000.

# <codecell>

RVShift

# <codecell>

file2_wl = RVT.extract_HERMES_wavelength(file2.filename()) + RVShift

# <codecell>

filei_wl = RVT.extract_iraf_wavelength(filei[0].header, 173)

# <codecell>

filep_wl = RVT.extract_pyhermes_wavelength(filep.filename())

# <codecell>

filem_wl = filem[:,0] + RVShift

# <codecell>

file2_data = file2[0].data

# <codecell>

filei_data = filei[0].data

# <codecell>

filep_data = filep[0].data

# <codecell>

filem_data = filem[:,1]

# <codecell>

plt.plot(file2_wl, file2_data[175]/np.median(file2_data[175]), label = '2dfdr')
plt.plot(filei_wl, filei_data[172]/np.median(filei_data[172]), label = 'iraf')
plt.plot(filep_wl, filep_data/nanmedian(filep_data), label = 'pyhermes')
plt.plot(filem_wl, filem_data/nanmedian(filem_data), label = 'myherpy')
plt.legend(loc=0)
plt.show()

# <codecell>


