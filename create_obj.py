# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob
import os
import numpy as np
import pyfits as pf
import pickle
import toolbox

# import pandas

# <codecell>

class star():
    
    name = ''
    RA = 0
    Dec = 0
    Vmag = 0
    
    def __init__(self, name, mode = '2dfdr'):
        if mode == '2dfdr' :self.load_star_data(name = name)
        if mode == 'iraf' :self.load_star_data_iraf(name = name)
        
    def load_star_data(self, name):
        
#         stetson_df = pandas.read_pickle('stetson.pd')
        files = glob.glob('cam1/*.fits')
        for thisFile in files:
            thisFits = pf.open(thisFile)
            fibreTable = thisFits['FIBRES'].data
            idx = fibreTable.field('NAME').strip()==name
            if fibreTable[idx].shape[0] >0: #if there is any star in this file...
                starInfo = fibreTable[idx][0] 

                self.name = starInfo.field('NAME').strip()
                self.RA = np.rad2deg(starInfo.field('RA'))
                self.Dec = np.rad2deg(starInfo.field('DEC'))        
                self.RA_deg, self.RA_min, self.RA_sec = toolbox.dec2sex(self.RA)   
                self.Dec_deg, self.Dec_min, self.Dec_sec = toolbox.dec2sex(self.Dec)
                self.Vmag = starInfo.field('MAGNITUDE')
            
#                 self.B = stetson_df[stetson_df.target == self.name].B.values[0]
#                 self.I = stetson_df[stetson_df.target == self.name].I.values[0]
#                 self.R = stetson_df[stetson_df.target == self.name].R.values[0]
#                 self.U = stetson_df[stetson_df.target == self.name].U.values[0]
#                 self.V = stetson_df[stetson_df.target == self.name].mag.values[0]
#                 self.BV = stetson_df[stetson_df.target == self.name].BV.values[0]
                print self.name,'star created'
                break
                
    def load_star_data_iraf(self, name):
        
#         stetson_df = pandas.read_pickle('stetson.pd')
        files = glob.glob('cam1/*.fits')
#         for thisFile in files:
        thisFits = pf.open(files[0])
        starNames = []
        for fib in thisFits[0].header['APID*']:
            if not (('PARKED' in fib.value) or ('Grid_Sky' in fib.value) or ('FIBRE ' in fib.value)): 
                if fib.value.split(' ')[0] == name:


                    self.name = name
#                         self.RA = np.rad2deg(starInfo.field('RA'))
#                         self.Dec = np.rad2deg(starInfo.field('DEC'))        
#                         self.RA_deg, self.RA_min, self.RA_sec = toolbox.dec2sex(self.RA)   
#                         self.Dec_deg, self.Dec_min, self.Dec_sec = toolbox.dec2sex(self.Dec)
                    self.Vmag = fib.value.split(' ')[1]

    #                 self.B = stetson_df[stetson_df.target == self.name].B.values[0]
    #                 self.I = stetson_df[stetson_df.target == self.name].I.values[0]
    #                 self.R = stetson_df[stetson_df.target == self.name].R.values[0]
    #                 self.U = stetson_df[stetson_df.target == self.name].U.values[0]
    #                 self.V = stetson_df[stetson_df.target == self.name].mag.values[0]
    #                 self.BV = stetson_df[stetson_df.target == self.name].BV.values[0]
                    print self.name,'star created'
                    break
                

# <codecell>

class camera():
    
    red_fluxes =  []
    wavelengths = []
#     CCCurves = []
#     clean_fluxes = []
#     clean_wavelengths  = []
#     Ps = []
#     shfted_wavelengths = []
    sigmas = []
    RVs = []
    fileNames = []
    Qs = []
#     safe_flag = []
    max_wl_range = []
    
    def __init__(self):
        self.red_fluxes =  []
        self.wavelengths = []
#         self.CCCurves = []
#         self.clean_fluxes = []
#         self.clean_wavelengths  = []
#         self.Ps = []
#         self.shfted_wavelengths = []
        self.sigmas = []
        self.RVs = []
        self.fileNames = []
        self.Qs = []
#         self.safe_flag = []
        self.max_wl_range = []

# <codecell>

import sys

# <codecell>

class exposures():

    def __init__(self):
#         self.UTdates = []
#         self.UTstarts = []
#         self.UTends = []
#         self.lengths = []
#         self.JDs = []
#         self.HRs = []
#         self.plates = []
#         self.pivots = []
        a = camera()
        b = camera()
        c = camera()
        d = camera()
        self.cameras = np.array([a,b,c,d])
        
    def load_exposures(self, name):
        
        print 'Collecting MJDs from all 4 channels'
        for camIdx, cam in enumerate(self.cameras):
            files = glob.glob('cam'+str(camIdx+1)+'/*.fits')
            print 'cam'+str(camIdx+1)+':',len(files) #,files
            for thisFile in files:    
                HDUList = pf.open(thisFile)
                e = float(HDUList[0].header['EXPOSED'])/2/24/60/60 # EXPOSED/2 in days
                inDate = np.hstack((HDUList[0].header['UTDATE'].split(':'),HDUList[0].header['UTSTART'].split(':'))).astype(int)
                MJD = toolbox.gd2jd(inDate, TZ=0) - 2400000.5 + e
                self.JDs.append(MJD)
        a = np.unique(np.round(np.array(self.JDs).astype(float),3))
        self.JDs = np.sort(a)
        nExposures = len(a)
        print nExposures, 'exposures per channel'
        print ''
        
        #create top_level arrays
        self.UTdates = np.chararray(nExposures,10)
        self.UTstarts = np.chararray(nExposures,10)
        self.UTends = np.chararray(nExposures,10)
        self.lengths = np.zeros(nExposures)
        self.plates = np.chararray(nExposures,10)
        self.pivots = np.zeros(nExposures).astype(int)
        self.HRs = np.zeros(nExposures).astype(bool)
        
        
        for camIdx, cam in enumerate(self.cameras):
            files = glob.glob('cam'+str(camIdx+1)+'/*.fits')
            print 'cam'+str(camIdx+1),len(files),'files'
            
            thisCam = cam
            
            #create camera level arrays
            thisCam.red_fluxes = np.zeros((nExposures, 4096))
            thisCam.wavelengths = np.zeros((nExposures, 4096))
            thisCam.fileNames = np.chararray(nExposures,18)
            thisCam.fileNames[:] =  ''

            
            for thisFile in files: 
                print 'Opening',thisFile,
                HDUList = pf.open(thisFile)
                fibreTable = HDUList['FIBRES'].data            
                idx = fibreTable.field('NAME').strip()==name
                if np.sum(idx)>0:  #star found in fits file 
                    
                    #one time per exposure (because they are equal in all cameras)
                    thisMJDidx = ''
                    e = float(HDUList[0].header['EXPOSED'])/2/24/60/60 # EXPOSED/2 in days
                    inDate = np.hstack((HDUList[0].header['UTDATE'].split(':'),HDUList[0].header['UTSTART'].split(':'))).astype(int)
                    MJD = toolbox.gd2jd(inDate, TZ=0) - 2400000.5 + e

                    thisMJDidx = np.where(self.JDs==round(float(MJD),3))[0]
                    if len(thisMJDidx)>0:
                        thisMJDidx = thisMJDidx[0]
                        print 'MJD',self.JDs[thisMJDidx]                          
                        self.UTdates[thisMJDidx] = HDUList[0].header['UTDATE']
                        self.UTstarts[thisMJDidx] = HDUList[0].header['UTSTART']
                        self.UTends[thisMJDidx] = HDUList[0].header['UTEND']
                        try:self.lengths[thisMJDidx] = HDUList[0].header['EXPOSED']
                        except: pass
#                         self.JDs.append(HDUList[0].header['UTMJD'])
                        self.plates[thisMJDidx] = HDUList[0].header['SOURCE']
                        self.pivots[thisMJDidx] = int(fibreTable.field('PIVOT')[idx][0])
                        print 'pivot',fibreTable.field('PIVOT')[idx]
                        if HDUList[0].header['SLITMASK'].strip()=='OUT':
                            self.HRs[thisMJDidx] = False
                        else:
                            self.HRs[thisMJDidx] = True
                                  
                        thisCam.red_fluxes[thisMJDidx] = HDUList[0].data[idx][0]
                        thisCam.wavelengths[thisMJDidx] = self.extract_HERMES_wavelength(HDUList[0].header)
                        thisCam.fileNames[thisMJDidx] = thisFile.split('/')[-1]
                    else:
                        print name, 'found in', thisFile, 'but no matching date' 
                else:
                    print name,'not found in', thisFile

            thisCam.safe_flag = np.ones(len(thisCam.fileNames)).astype(bool)
            print ''

    def load_exposures_iraf(self, name):
        
        print 'Collecting MJDs from all 4 channels'
        for camIdx, cam in enumerate(self.cameras):
            files = glob.glob('cam'+str(camIdx+1)+'/*.fits')
            print 'cam'+str(camIdx+1)+':',len(files) #,files
            for thisFile in files:    
                HDUList = pf.open(thisFile)
                e = float(HDUList[0].header['EXPOSED'])/2/24/60/60 # EXPOSED/2 in days
                inDate = np.hstack((HDUList[0].header['UTDATE'].split(':'),HDUList[0].header['UTSTART'].split(':'))).astype(int)
                MJD = toolbox.gd2jd(inDate, TZ=0) - 2400000.5 + e
                self.JDs.append(MJD)
        a = np.unique(np.round(np.array(self.JDs).astype(float),3))
        self.JDs = np.sort(a)
        nExposures = len(a)
        print nExposures, 'exposures per channel'
        print ''
        
        #create top_level arrays
        self.UTdates = np.chararray(nExposures,10)
        self.UTstarts = np.chararray(nExposures,10)
        self.UTends = np.chararray(nExposures,10)
        self.lengths = np.zeros(nExposures)
        self.plates = np.chararray(nExposures,10)
        self.pivots = np.zeros(nExposures).astype(int)
        self.HRs = np.zeros(nExposures).astype(bool)
        
        
        for camIdx, cam in enumerate(self.cameras):
            files = glob.glob('cam'+str(camIdx+1)+'/*.fits')
            print 'cam'+str(camIdx+1),len(files),'files'
            
            thisCam = cam
            
            #create camera level arrays
            thisCam.red_fluxes = np.zeros((nExposures, 4096))
            thisCam.wavelengths = np.zeros((nExposures, 4096))
            thisCam.fileNames = np.chararray(nExposures,18)
            thisCam.fileNames[:] =  ''

            
            for thisFile in files: 
                print 'Opening',thisFile,
                HDUList = pf.open(thisFile)
                starNames = []
                fits = pf.open(thisFile)
                for fib in fits[0].header['APID*']:
                    if fib.value.split(' ')[0]==name:
                        print name,'found in', thisFile,'pivot =' ,fib.key[4:]

                        #one time per exposure (because they are equal in all cameras)
                        thisMJDidx = ''
                        e = float(HDUList[0].header['EXPOSED'])/2/24/60/60 # EXPOSED/2 in days
                        inDate = np.hstack((HDUList[0].header['UTDATE'].split(':'),HDUList[0].header['UTSTART'].split(':'))).astype(int)
                        MJD = toolbox.gd2jd(inDate, TZ=0) - 2400000.5 + e

                        thisMJDidx = np.where(self.JDs==round(float(MJD),3))[0]
                        if len(thisMJDidx)>0:
                            thisMJDidx = thisMJDidx[0]
                            print 'MJD',self.JDs[thisMJDidx]                          
                            self.UTdates[thisMJDidx] = HDUList[0].header['UTDATE']
                            self.UTstarts[thisMJDidx] = HDUList[0].header['UTSTART']
                            self.UTends[thisMJDidx] = HDUList[0].header['UTEND']
                            try:self.lengths[thisMJDidx] = HDUList[0].header['EXPOSED']
                            except: pass
    #                         self.JDs.append(HDUList[0].header['UTMJD'])
                            self.plates[thisMJDidx] = HDUList[0].header['SOURCE']
                            self.pivots[thisMJDidx] = int(fib.key[4:])
                            if HDUList[0].header['SLITMASK'].strip()=='OUT':
                                self.HRs[thisMJDidx] = False
                            else:
                                self.HRs[thisMJDidx] = True
                            if thisMJDidx>0: #interpolates to match epoch 0
                            
                                initialFlux = HDUList[0].data[self.pivots[thisMJDidx]]
                                initialWL = self.extract_IRAF_wavelength(HDUList[0].header, self.pivots[thisMJDidx])
                                refWL = thisCam.wavelengths[0]
                                
                                thisCam.red_fluxes[thisMJDidx] = np.interp(refWL, initialWL, initialFlux) 
                                thisCam.wavelengths[thisMJDidx] = thisCam.wavelengths[0]
                            else:
                                thisCam.red_fluxes[thisMJDidx] = HDUList[0].data[self.pivots[thisMJDidx]]
                                thisCam.wavelengths[thisMJDidx] = self.extract_IRAF_wavelength(HDUList[0].header, self.pivots[thisMJDidx])
                                thisCam.red_fluxes[thisMJDidx] = np.interp(thisCam.wavelengths[thisMJDidx], thisCam.wavelengths[thisMJDidx], thisCam.red_fluxes[thisMJDidx]) 

                            thisCam.fileNames[thisMJDidx] = thisFile.split('/')[-1]
                        else:
                            print name, 'found in', thisFile, 'but no matching date' 
#                     else:
#                         print name,'not found in', thisFile

            thisCam.safe_flag = np.ones(len(thisCam.fileNames)).astype(bool)
            print ''

            
    def extract_HERMES_wavelength(self, header):
        
        CRVAL1 = header['CRVAL1'] # / Co-ordinate value of axis 1                    
        CDELT1 = header['CDELT1'] #  / Co-ordinate increment along axis 1             
        CRPIX1 = header['CRPIX1'] #  / Reference pixel along axis 1                   
        
        #Creates an array of offset wavelength from the referece px/wavelength
        Lambda = CRVAL1 - (CRPIX1 - (np.arange(int(CRPIX1)*2)) -1)* CDELT1
    
        return Lambda

    def extract_IRAF_wavelength(self, header, app):

        WS = 'WS_'+str(int(app))
        WD = 'WD_'+str(int(app))

        first_px = float(header[WS])
        disp = float(header[WD])
        length = header['NAXIS1']
        wl = np.arange(length)*disp
        wl += first_px

        return wl

    
    def calculate_baryVels(self, star):
        baryVels = []
        for j in self.JDs:
            vh, vb = toolbox.baryvel(j+2400000+0.5) 
            ra = star.RA    #RA  in degress
            dec = star.Dec  #Dec in degrees
            baryVels.append((vb[0]*np.cos(np.radians(dec))*np.cos(np.radians(ra)) + vb[1]*np.cos(np.radians(dec))*np.sin(np.radians(ra)) + vb[2]*np.sin(np.radians(dec)))*1000)
#         print baryVels
        self.rel_baryVels = np.array(baryVels) - baryVels[0]
        self.abs_baryVels = np.array(baryVels)

        

