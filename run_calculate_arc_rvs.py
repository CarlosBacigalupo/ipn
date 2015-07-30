#!/usr/bin/python

import glob
import os
import importlib
import numpy as np
import sys
import pyfits as pf
import pylab as plt
import RVTools as RVT
from scipy import signal, optimize, constants


booCopy = False
booPlot = False
medianRange = 5 
corrHWidth = 5
xDef = 1
#     fibres = [75]
#     fibres = [7]
fibres = range(400)

if len(sys.argv)>1:
    dataset = sys.argv[1]
    try:
        thisDataset = importlib.import_module('data_sets.'+dataset)
    except:
        print 'Could not load',dataset         
        sys.exit()


    #Copy all files
    if booCopy==True:
            
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
                strCopy += 'arc_cam'+ str(cam) + '/' + filename_prfx[i] + str(cam) + thisFile + 'red.fits ' 
        #         strCopy += '.' 
                print strCopy
                os.system(strCopy)

            
    #Create arcRVs
    os.chdir('/Users/Carlos/Documents/HERMES/reductions/6.5/'+dataset+'/')
    
    arcRVs = np.ones((400,5,4))*np.nan
    
    if ((len(fibres)>5) and (booPlot==True)): 
        print 'Too many plots, quitting. You are welcome.'
        sys.exit()
        
    for cam in range(4):
        for fibre in fibres: 
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
    
                lambda1, flux1 = RVT.clean_flux(refWL, refData[fibre], flatten = False, medianRange = medianRange)
                if booPlot==True:plt.plot(lambda1,flux1)
                lambda2, flux2 = RVT.clean_flux(thisWL, thisData[fibre], flatten = False, medianRange = medianRange)
                if booPlot==True:plt.plot(lambda2,flux2)
                if booPlot==True:plt.title(str(i)+' '+thisFile)
    
                try:
                    CCCurve = signal.fftconvolve(flux1[-np.isnan(flux1)], flux2[-np.isnan(flux2)][::-1], mode='same')
                    print 'Flux1 nans', sum(np.isnan(flux1))
                    print 'Flux2 nans', sum(np.isnan(flux2))
                    if booPlot==True:plt.show()
                    if booPlot==True:plt.plot(CCCurve)
                    corrMax = np.where(CCCurve==max(CCCurve))[0][0]
                    p_guess = [corrMax,corrHWidth]
                    x_mask = np.arange(corrMax-corrHWidth, corrMax+corrHWidth+1)
                    if max(x_mask)<len(CCCurve):
                        p = RVT.fit_gaussian(p_guess, CCCurve[x_mask], np.arange(len(CCCurve))[x_mask])[0]
                        if np.modf(CCCurve.shape[0]/2.0)[0]>1e-5:
                            pixelShift = (p[0]-(CCCurve.shape[0]-1)/2.) #odd number of elements
                        else:
                            pixelShift = (p[0]-(CCCurve.shape[0])/2.) #even number of elements
    
    
                        mid_px = refData.shape[1]/2
                        dWl = (refWL[mid_px+1]-refWL[mid_px]) / refWL[mid_px]/xDef
                        print 'Parameters CCCurve.shape, corrMax, p_guess, p, pixelshift, mid_px, dWl'
                        print CCCurve.shape, p_guess, p, pixelShift, mid_px, dWl
                        RV = dWl * pixelShift * constants.c 
                        print 'RV',fibre,i,RV
                        arcRVs[fibre,i,cam] = RV
                    if booPlot==True:plt.title('RV '+str(RV))
                    if booPlot==True:plt.show()
    
                except Exception,e: 
                    print str(e)
                    plt.close()
    
    #hack to fix epoch range 
    if dataset=='HD1581':
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
        
    elif dataset=='HD285507':
        print arcRVs.shape
        arcRVs2 = np.ones((400,15,4))*np.nan
        arcRVs2[:,0,:] = arcRVs[:,0,:]
        arcRVs2[:,1,:] = arcRVs[:,0,:]
        arcRVs2[:,2,:] = arcRVs[:,0,:]
        arcRVs2[:,3,:] = arcRVs[:,1,:]
        arcRVs2[:,4,:] = arcRVs[:,1,:]
        arcRVs2[:,5,:] = arcRVs[:,1,:]
        arcRVs2[:,6,:] = arcRVs[:,2,:]
        arcRVs2[:,7,:] = arcRVs[:,2,:]
        arcRVs2[:,8,:] = arcRVs[:,2,:]
        arcRVs2[:,9,:] = arcRVs[:,3,:]
        arcRVs2[:,10,:] = arcRVs[:,3,:]
        arcRVs2[:,11,:] = arcRVs[:,3,:]
        arcRVs2[:,12,:] = arcRVs[:,4,:]
        arcRVs2[:,13,:] = arcRVs[:,4,:]
        arcRVs2[:,14,:] = arcRVs[:,4,:]
        arcRVs = arcRVs2
        print arcRVs.shape
    
    if len(fibres)==400: np.save('npy/arcRVs',arcRVs)
else:
    print 'no dataset specified'