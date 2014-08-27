import HERMES
import pylab as plt
import numpy as np
import glob 

import os
import subprocess 
import toolbox
import scipy.optimize as opt

a = HERMES.PSF()

a.sexParamFile = 'HERMES.sex'
a.sex_path = '/usr/local/bin/'
a.outputFileName = 'out.txt'
# a.nFibres = 10

# a.scienceFile = '07feb' + str(a.camera) + '0022.fits'
# a.biasFile = 'BIAScombined2.fits'
a.profile = 'gaussian'
files = glob.glob('*.fits')
for file in files:
    a.camera = 4
    a.arcFile = file
    a.read_full_image_spectral('gaussian')
