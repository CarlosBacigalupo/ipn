
# coding: utf-8

# In[2]:

import glob
import os
import numpy as np
import pyfits as pf
import pickle
import toolbox

# import pandas


# In[5]:

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
                


# In[6]:

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


# In[1]:

import sys


# In[1]:

class exposures():

    def __init__(self):
#         self.UTdates = []
#         self.UTstarts = []
#         self.UTends = []
#         self.lengths = []
        self.MJDs = []
#         self.HRs = []
#         self.plates = []
#         self.pivots = []
        a = camera()
        b = camera()
        c = camera()
        d = camera()
        self.cameras = np.array([a,b,c,d])
        
    def load_multi_exposures(self, name, pivot, plate, MJD):
        
#         print 'Collecting MJDs from all 4 channels'
#         for camIdx, cam in enumerate(self.cameras):
#             files = glob.glob('cam'+str(camIdx+1)+'/*.fits')
#             print 'cam'+str(camIdx+1)+':',len(files) #,files
#             for thisFile in files:    
#                 HDUList = pf.open(thisFile)
#                 e = float(HDUList[0].header['EXPOSED'])/2/24/60/60 # EXPOSED/2 in days
#                 inDate = np.hstack((HDUList[0].header['UTDATE'].split(':'),HDUList[0].header['UTSTART'].split(':'))).astype(int)
#                 MJD = toolbox.gd2jd(inDate, TZ=0) - 2400000.5 + e
#                 self.JDs.append(MJD)
#         a = np.unique(np.round(np.array(self.JDs).astype(float),3))
#         self.JDs = np.sort(a)
#         nExposures = len(a)
#         print nExposures, 'exposures per channel'
#         print ''
        
        #create top_level empty arrays
        self.MJDs = []
        self.UTdates = []
        self.UTstarts = []
        self.UTends = []
        self.lengths = []
        self.plates = []
        self.pivots = []
        self.HRs = []
        


        thisCam = self.cameras[0]
        camIdx = 0 
        
        #create camera level arrays
        thisCam.red_fluxes = []
        thisCam.wavelengths = []
        thisCam.fileNames = []


        files = glob.glob('cam'+str(camIdx+1)+'/*.fits')
        for thisFile in files: 
            print 'Opening',thisFile
            HDUList = pf.open(thisFile)
            fibreTable = HDUList['FIBRES'].data
            
            booIdx = fibreTable.field('NAME').strip()==name
            if np.sum(booIdx)>0:  #star found in fits file 
                thisPlate = int(HDUList[0].header['SOURCE'][-1])
                thisPivot = int(fibreTable.field('PIVOT')[booIdx][0])

                e = float(HDUList[0].header['EXPOSED'])/2/24/60/60 # EXPOSED/2 in days
                inDate = np.hstack((HDUList[0].header['UTDATE'].split(':'),HDUList[0].header['UTSTART'].split(':'))).astype(int)
                thisMJD = toolbox.gd2jd(inDate, TZ=0) - 2400000.5 + e
                
                #debug stuff
#                 print 'filter:', pivot, plate, MJD
#                 print 'filter:', type(pivot), type(plate), type(MJD)
#                 print 'this file:',thisPivot, thisPlate, thisMJD
#                 print 'this file:', type(thisPivot), type(thisPlate), type(thisMJD)
                
                #do I want this data?
                if ((pivot==thisPivot) and (plate==thisPlate) and (round(thisMJD,5)>=round(MJD,5))):
                    print 'Valid Data point. Updating...'
                    print
                    
                    #exposure level (only cam1)
                    self.MJDs.append(thisMJD)
                    self.UTdates.append(HDUList[0].header['UTDATE'])
                    self.UTstarts.append(HDUList[0].header['UTSTART'])
                    self.UTends.append(HDUList[0].header['UTEND'])
                    try:
                        self.lengths.append(HDUList[0].header['EXPOSED'])
                    except:
                        self.lengths.append(np.nan)
                    self.plates.append(thisPlate)
                    self.pivots.append(thisPivot)
                    if HDUList[0].header['SLITMASK'].strip()=='OUT':
                        self.HRs.append(False)
                    else:
                        self.HRs.append(True)
            
                    thisCam.red_fluxes.append(HDUList[0].data[booIdx][0])
                    thisCam.wavelengths.append(self.extract_HERMES_wavelength(HDUList[0].header))
                    thisCam.fileNames.append(thisFile.split('/')[-1])

        #after loading all cam1, order them by MJD
        #first convert them into numpy
        self.MJDs = np.array(self.MJDs)
        self.UTdates = np.array(self.UTdates)
        self.UTstarts = np.array(self.UTstarts)
        self.UTends = np.array(self.UTends)
        self.lengths = np.array(self.lengths)
        self.plates = np.array(self.plates)
        self.pivots = np.array(self.pivots)
        self.HRs = np.array(self.HRs)        
        
        thisCam.red_fluxes = np.array(thisCam.red_fluxes)        
        thisCam.wavelengths = np.array(thisCam.wavelengths)        
        thisCam.fileNames = np.array(thisCam.fileNames)        

        #get the order by MJD
        order = np.argsort(self.MJDs)

        #order exposure level 
        self.MJDs = self.MJDs[order]
        self.UTdates = self.UTdates[order]
        self.UTstarts = self.UTstarts[order]
        self.UTends = self.UTends[order]
        self.lengths = self.lengths[order]
        self.plates = self.plates[order]
        self.pivots = self.pivots[order]
        self.HRs = self.HRs[order]
        
        #order camera level 
        thisCam.red_fluxes = thisCam.red_fluxes[order]
        thisCam.wavelengths = thisCam.wavelengths[order]
        thisCam.fileNames = thisCam.fileNames[order]

        #test exposure level results
#         print self.MJDs
#         print self.UTdates
#         print self.UTstarts
#         print self.UTends
#         print self.lengths
#         print self.plates
#         print self.pivots
#         print self.HRs
        
        #test camera level results
#         print thisCam.red_fluxes
#         print thisCam.wavelengths
#         print thisCam.fileNames
        
        print 
        print 'Starting cameras 2-4'
        for camIdx, thisCam in enumerate(self.cameras[1:]):  
            print'>>>>>New Star.........' 
            print
            
            #create camera level arrays
            ordMJD = []   #holds the indices found in cam1 to order loaded flux, wl and filenames array at the end. 
            thisCam.red_fluxes = []
            thisCam.wavelengths = []
            thisCam.fileNames = []

            files = glob.glob('cam'+str(camIdx+2)+'/*.fits') # +2 because camIdx is 1 index behind due to [1:] filter in for loop
            for thisFile in files: 
                print 'Opening',thisFile
                HDUList = pf.open(thisFile)
                fibreTable = HDUList['FIBRES'].data
            
                booIdx = fibreTable.field('NAME').strip()==name
                if np.sum(booIdx)>0:  #star found in fits file 
                    thisPlate = int(HDUList[0].header['SOURCE'][-1])
                    thisPivot = int(fibreTable.field('PIVOT')[booIdx][0])

                    e = float(HDUList[0].header['EXPOSED'])/2/24/60/60 # EXPOSED/2 in days
                    inDate = np.hstack((HDUList[0].header['UTDATE'].split(':'),HDUList[0].header['UTSTART'].split(':'))).astype(int)
                    thisMJD = toolbox.gd2jd(inDate, TZ=0) - 2400000.5 + e

#                     print 'filter:', pivot, plate, MJD
#                     print 'filter:', type(pivot), type(plate), type(MJD)
#                     print 'this file:',thisPivot, thisPlate, thisMJD
#                     print 'this file:', type(thisPivot), type(thisPlate), type(thisMJD)
                
                    #do I want this data?
                    if ((pivot==thisPivot) and (plate==thisPlate) and (round(thisMJD,5)>=round(MJD,5))):
#                         print round(thisMJD,5)>=round(MJD,5)
                        print 'Valid Data point. Updating...'
#                         print self.MJDs
                        MJDIdx = np.where(np.round(self.MJDs,3)==np.round(thisMJD,3))[0]
                        print MJDIdx,
                        if MJDIdx.shape[0]>1:
                            print 'Too many MJD indices returned. Aborting!'
                            raise SystemExit(0)
                        elif MJDIdx.shape[0]==0:
                            print 'MJD index NOT found. Aborting!'
                            print 'Looking for',thisMJD,'in',self.MJDs
                            raise SystemExit(0)
                        elif MJDIdx.shape[0]==1:
                            print 'MJD index',MJDIdx[0],'found. Updating...'
                            MJDIdx = MJDIdx[0]
                            
                            ordMJD.append(MJDIdx)
                            thisCam.red_fluxes.append(HDUList[0].data[booIdx][0])
                            thisCam.wavelengths.append(self.extract_HERMES_wavelength(HDUList[0].header))
                            thisCam.fileNames.append(thisFile.split('/')[-1])
                        print 
        
        
            #convert the order into indices
            order = np.ones(self.MJDs.shape[0]) * np.nan
            for i,idx in enumerate(ordMJD):
                order[idx] = int(i)
            
            #if nans left complete the missing days
            nNans = np.sum(np.isnan(order))
            print 'Number of NaNs in order:',nNans
            if nNans>0:
                print np.isnan(order),order.shape[0]
                for i in np.arange(order.shape[0])[np.isnan(order)]:
                    print 'Fixing NaNs in',i
                    order[i] = len(thisCam.red_fluxes) #replaces NaN in order for value of length (i.e. the nans it's about to add)
                    thisCam.red_fluxes.append(np.ones(thisCam.red_fluxes[0].shape[0])*np.nan)
                    thisCam.wavelengths.append(np.ones(thisCam.red_fluxes[0].shape[0])*np.nan)
                    thisCam.fileNames.append('')
                
            order = order.astype(int)
            
            #after loading all this cam, order them by MJD
            #first convert them into numpy
            thisCam.red_fluxes = np.array(thisCam.red_fluxes)        
            thisCam.wavelengths = np.array(thisCam.wavelengths)        
            thisCam.fileNames = np.array(thisCam.fileNames)        

            #order camera level 
            thisCam.red_fluxes = thisCam.red_fluxes[order]
            thisCam.wavelengths = thisCam.wavelengths[order]
            thisCam.fileNames = thisCam.fileNames[order]
        
            #test camera level results
#             print ordMJD, order
#             print thisCam.red_fluxes
#             print thisCam.wavelengths
#             print thisCam.fileNames
    
        print 
            
            
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
                self.MJDs.append(MJD)
        a = np.unique(np.round(np.array(self.MJDs).astype(float),3))
        self.MJDs = np.sort(a)
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

                    thisMJDidx = np.where(self.MJDs==round(float(MJD),3))[0]
                    if len(thisMJDidx)>0:
                        thisMJDidx = thisMJDidx[0]
                        print 'MJD',self.MJDs[thisMJDidx]                          
                        self.UTdates[thisMJDidx] = HDUList[0].header['UTDATE']
                        self.UTstarts[thisMJDidx] = HDUList[0].header['UTSTART']
                        self.UTends[thisMJDidx] = HDUList[0].header['UTEND']
                        try:self.lengths[thisMJDidx] = HDUList[0].header['EXPOSED']
                        except: pass
#                         self.MJDs.append(HDUList[0].header['UTMJD'])
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

            
    def load_exposures_myherpy(self, name, camIdx):
        
        print 'Collecting MJDs from camera', camIdx
        
        WLwildcard = 'cam'+str(camIdx+1)+'/WLS_F*.npy' 
        FLUXwildcard = 'cam'+str(camIdx+1)+'/extracted_obj*.npy' 
#         wildcard = 'cam'+str(camIdx+1)+'/HD1581*.txt'
#         print wildcard
        WLfiles = glob.glob(WLwildcard)
        FLUXfiles = glob.glob(FLUXwildcard)
        print 'cam'+str(camIdx+1)+':',len(files) #,files    
        thisCam = self.cameras[camIdx]
        for thisFileIdx in range(len(files)): 
            print 'Opening',WLfiles[thisFileIdx], 'and',FLUXfiles[thisFileIdx],
            rawtxt = np.loadtxt(thisFile)
            self.MJDs.append(int(thisFile.split('.')[1])/1000.)
            thisCam.wavelengths.append(rawtxt[:,0]) 
            thisCam.red_fluxes.append(rawtxt[:,1]) 
        
        self.MJDs = np.array(self.MJDs)
        thisCam.wavelengths = np.array(thisCam.wavelengths)
        thisCam.red_fluxes = np.array(thisCam.red_fluxes)
            
            
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
                self.MJDs.append(MJD)
        a = np.unique(np.round(np.array(self.MJDs).astype(float),3))
        self.MJDs = np.sort(a)
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

                        thisMJDidx = np.where(self.MJDs==round(float(MJD),3))[0]
                        if len(thisMJDidx)>0:
                            thisMJDidx = thisMJDidx[0]
                            print 'MJD',self.MJDs[thisMJDidx]                          
                            self.UTdates[thisMJDidx] = HDUList[0].header['UTDATE']
                            self.UTstarts[thisMJDidx] = HDUList[0].header['UTSTART']
                            self.UTends[thisMJDidx] = HDUList[0].header['UTEND']
                            try:self.lengths[thisMJDidx] = HDUList[0].header['EXPOSED']
                            except: pass
    #                         self.MJDs.append(HDUList[0].header['UTMJD'])
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
        for j in self.MJDs:
            vh, vb = toolbox.baryvel(j+2400000+0.5) 
            ra = star.RA    #RA  in degress
            dec = star.Dec  #Dec in degrees
            baryVels.append((vb[0]*np.cos(np.radians(dec))*np.cos(np.radians(ra)) + vb[1]*np.cos(np.radians(dec))*np.sin(np.radians(ra)) + vb[2]*np.sin(np.radians(dec)))*1000)
#         print baryVels
        self.rel_baryVels = np.array(baryVels) - baryVels[0]
        self.abs_baryVels = np.array(baryVels)

        

