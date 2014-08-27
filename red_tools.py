# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob
import os
import numpy as np
import pyfits as pf
# from scipy import signal, interpolate, optimize, constants
# import pylab as plt
import pickle
# import TableBrowser as TB
import toolbox
import pandas

# <codecell>

os.chdir('/Users/Carlos/Documents/HERMES/reductions/m67_all_4cams')

# <codecell>

class star():
    
    name = ''
    RA_dec = 0
    Dec_dec = 0
    Vmag = 0
    
    def __init__(self, name):
        self.load_star_data(name = name)
        
        
    def load_star_data(self, name):
        
        stetson_df = pandas.read_pickle('stetson.pd')
        files = glob.glob('cam1/*red.fits')
        for thisFile in files:
            a = pf.open(thisFile)
            b = a['FIBRES'].data
            idx = b.field('NAME').strip()==name
            
            if b[idx].shape[0] >0:
                starInfo = b[idx][0]
                
                self.name = starInfo.field('NAME').strip()
                self.RA_dec = starInfo.field('RA')
                self.Dec_dec = starInfo.field('DEC')        
                self.RA_h, self.RA_min, self.RA_sec = toolbox.dec2sex(self.RA_dec/15)   
                self.Dec_deg, self.Dec_min, self.Dec_sec = toolbox.dec2sex(self.Dec_dec)
                self.Vmag = starInfo.field('MAGNITUDE')
            
                self.B = stetson_df[stetson_df.target == self.name].B.values[0]
                self.I = stetson_df[stetson_df.target == self.name].I.values[0]
                self.R = stetson_df[stetson_df.target == self.name].R.values[0]
                self.U = stetson_df[stetson_df.target == self.name].U.values[0]
                self.V = stetson_df[stetson_df.target == self.name].mag.values[0]
                self.BV = stetson_df[stetson_df.target == self.name].BV.values[0]
                print self.name,'star created'
                break
                

# <codecell>

class camera():
    
    red_fluxes =  []
    wavelengths = []
    CCCurves = []
    clean_fluxes = []
    clean_wavelengths  = []
    Ps = []
    shfted_wavelengths = []
    sigmas = []
    RVs = []
    fileNames = []
    safe_flag = []
    
    def __init__(self):
        self.red_fluxes =  []
        self.wavelengths = []
        self.CCCurves = []
        self.clean_fluxes = []
        self.clean_wavelengths  = []
        self.Ps = []
        self.shfted_wavelengths = []
        self.sigmas = []
        self.RVs = []
        self.fileNames = []
        self.safe_flag = []
        
        

# <codecell>

class exposures():

    def __init__(self):
        self.UTdates = []
        self.UTstarts = []
        self.UTends = []
        self.lengths = []
        self.JDs = []
        self.HRs = []
        self.plates = []
        self.pivots = []
        a = camera()
        b = camera()
        c = camera()
        d = camera()
        self.cameras = np.array([a,b,c,d])
        
    def load_exposures(self, name):

        for camIdx in range(4):
            files = glob.glob('cam'+str(camIdx+1)+'/*red.fits')    
            thisCam = self.cameras[camIdx]
            for thisFile in files:    
                
                HDUList = pf.open(thisFile)
                fibreTable = HDUList['FIBRES'].data            
                idx = fibreTable.field('NAME').strip()==name

                if np.sum(idx)>0:  #star found in fits file 
                    if camIdx == 0: #one time per exposure (because they are equal in all cameras)
                        self.UTdates.append(HDUList[0].header['UTDATE'])
                        self.UTstarts.append(HDUList[0].header['UTSTART'])
                        self.UTends.append(HDUList[0].header['UTEND'])
                        self.lengths.append(HDUList[0].header['EXPOSED'])
                        self.JDs.append(HDUList[0].header['UTMJD'])
                        self.plates.append(HDUList[0].header['SOURCE'])
                        self.pivots.append(fibreTable.field('PIVOT')[idx][0])
                        if HDUList[0].header['SLITMASK'].strip()=='OUT':
                            self.HRs.append(False)
                        else:
                            self.HRs.append(True)
                    thisCam.red_fluxes.append(HDUList[0].data[idx][0])
                    thisCam.wavelengths.append(self.extract_HERMES_wavelength(HDUList[0].header))
                    thisCam.fileNames.append(thisFile.split('/')[-1])
                    
                        

#             sortOrder = np.argsort(self.JDs)
#             print sortOrder
#             if camIdx == 3:
#                 self.UTdates = np.array(self.UTdates)[sortOrder]
#                 self.UTstarts = np.array(self.UTstarts)[sortOrder]
#                 self.UTends = np.array(self.UTends)[sortOrder]
#                 self.lengths = np.array(self.lengths)[sortOrder]
#                 self.JDs = np.array(self.JDs)[sortOrder]
#                 self.HRs = np.array(self.HRs)[sortOrder]
#                 self.plates = np.array(self.plates)[sortOrder]
#                 self.pivots = np.array(self.pivots)[sortOrder]

#             thisCam.red_fluxes = np.array(thisCam.red_fluxes)[sortOrder]
#             thisCam.wavelengths = np.array(thisCam.wavelengths)[sortOrder]
#             thisCam.fileNames = np.array(thisCam.fileNames)[sortOrder]
            self.UTdates = np.array(self.UTdates)
            self.UTstarts = np.array(self.UTstarts)
            self.UTends = np.array(self.UTends)
            self.lengths = np.array(self.lengths)
            self.JDs = np.array(self.JDs)
            self.HRs = np.array(self.HRs)
            self.plates = np.array(self.plates)
            self.pivots = np.array(self.pivots)
            thisCam.red_fluxes = np.array(thisCam.red_fluxes)
            thisCam.wavelengths = np.array(thisCam.wavelengths)
            thisCam.fileNames = np.array(thisCam.fileNames)
            thisCam.safe_flag = np.ones(len(thisCam.fileNames)).astype(bool)


            
    def extract_HERMES_wavelength(self, header):
        
        CRVAL1 = header['CRVAL1'] # / Co-ordinate value of axis 1                    
        CDELT1 = header['CDELT1'] #  / Co-ordinate increment along axis 1             
        CRPIX1 = header['CRPIX1'] #  / Reference pixel along axis 1                   
        
        #Creates an array of offset wavelength from the referece px/wavelength
        Lambda = CRVAL1 - (CRPIX1 - (np.arange(int(CRPIX1)*2)) -1)* CDELT1
    
        return Lambda

    
    def calculate_baryVels(self, star):
        baryVels = []
        for j in self.JDs:
            vh, vb = toolbox.baryvel(j+2400000+0.5) 
            ra = star.RA_dec    #RA  in radians
            dec = star.Dec_dec  #Dec in radians
            baryVels.append(-(vb[0]*np.cos(dec)*np.cos(ra) + vb[1]*np.cos(dec)*np.sin(ra) + vb[2]*np.sin(dec))*1000)
#         print baryVels
        self.red_baryVels = np.array(baryVels) - baryVels[0]
        self.abs_baryVels = np.array(baryVels)

        

# <codecell>

# data = np.load('data.npy')
# starList = data[:,[0,3]][np.argsort(data[:,3].astype(float))]

# <codecell>

# starList[:30,:]

# <codecell>

# for i in starList[:31,0]:
#     thisStar = star(i)
#     thisStar.exposures = exposures()
#     thisStar.exposures.load_exposures(thisStar.name)
#     thisStar.exposures.calculate_baryVels(thisStar)
#     file_pi = open(thisStar.name+'.obj', 'w') 
#     pickle.dump(thisStar, file_pi) 
#     file_pi.close()
#     thisStar = None

