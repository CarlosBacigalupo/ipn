# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob
import os
import numpy as np
import pyfits as pf
from scipy import signal, interpolate, optimize, constants
import pylab as plt
import pickle
import TableBrowser as TB
import toolbox
import pandas

# <codecell>

class star():
    
    name = ''
    RA_dec = 0
    Dec_dec = 0
    Vmag = 0
    
    def __init__(self, name, xDef = 1, booWrite = False):
        
        self.load_star_data(name = name)
        self.exposures = exposures()
        self.exposures.load_exposures(name = self.name, xDef = xDef) #class containing exposures for this star
        self.calculate_baryVels()
        
        if booWrite==True:
            file_pi = open(self.name+'.obj', 'w') 
            pickle.dump(self, file_pi) 

    def calculate_baryVels(self):
        baryVels = []
        for j in self.exposures.JDs:
            vh, vb = toolbox.baryvel(j+2400000+0.5) 
            ra = self.RA_dec    #RA  in radians
            dec = self.Dec_dec  #Dec in radians
            baryVels.append(-(vb[0]*np.cos(dec)*np.cos(ra) + vb[1]*np.cos(dec)*np.sin(ra) + vb[2]*np.sin(dec))*1000)
#         print baryVels
        self.baryVels = np.array(baryVels) - baryVels[0]
        self.absBaryVels = np.array(baryVels)


        
        
    def load_star_data(self, name):
        
        stetson_df = pandas.read_pickle('stetson.pd')
        os.chdir('cam1/')
        files = glob.glob('*.fits')
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

        os.chdir('..')



        

# <codecell>

class exposures():

    def __init__(self):
        self.UTdates = []
        self.UTstarts = []
        self.UTends = []
        self.lengths = []
        self.cam1_fluxes = []
        self.cam1_wavelengths = []
        self.cam1_fileNames = []
        self.cam1_RVs = RVs()
        self.cam2_fluxes = []
        self.cam2_wavelengths = []
        self.cam2_fileNames = []
        self.cam2_RVs = RVs()
        self.cam3_fluxes = []
        self.cam3_wavelengths = []
        self.cam3_fileNames = []
        self.cam3_RVs = RVs()
        self.cam4_fluxes = []
        self.cam4_wavelengths = []
        self.cam4_fileNames = []
        self.cam4_RVs = RVs()
        self.JDs = []
        self.HRs = []

        
    def load_exposures(self, name, xDef = 1):
        for thisCam in range(1,5):
            os.chdir('cam'+str(thisCam)+'/')
#             print os.curdir
            files = glob.glob('*.fits')    
            for thisFile in files:    
                
                HDUList = pf.open(thisFile)
                fibreTable = HDUList['FIBRES'].data            
                idx = fibreTable.field('NAME').strip()==name
                print thisFile, np.sum(idx)
    
                if np.sum(idx)>0:  #star found in fits file              
                    if thisCam == 1:
                        self.UTdates.append(HDUList[0].header['UTDATE'])
                        self.UTstarts.append(HDUList[0].header['UTSTART'])
                        self.UTends.append(HDUList[0].header['UTEND'])
                        self.lengths.append(HDUList[0].header['EXPOSED'])
                        self.JDs.append(HDUList[0].header['UTMJD'])
                        if HDUList[0].header['SLITMASK'].strip()=='OUT':
                            self.HRs.append(False)
                        else:
                            self.HRs.append(True)
                        self.cam1_fluxes.append(HDUList[0].data[idx][0])
                        self.cam1_wavelengths.append(self.extract_HERMES_wavelength(HDUList[0].header))
                        self.cam1_fileNames.append(thisFile)
#                         print thisCam, np.mean(self.cams[thisCam - 1].wavelengths[0])    

                    elif thisCam == 2:
                        self.cam2_fluxes.append(HDUList[0].data[idx][0])
                        self.cam2_wavelengths.append(self.extract_HERMES_wavelength(HDUList[0].header))
                        self.cam2_fileNames.append(thisFile)
#                         print thisCam, np.mean(self.cams[thisCam - 1].wavelengths[0])    
                                                   
                    elif thisCam == 3:
                        self.cam3_fluxes.append(HDUList[0].data[idx][0])
                        self.cam3_wavelengths.append(self.extract_HERMES_wavelength(HDUList[0].header))
                        self.cam3_fileNames.append(thisFile)
#                         print thisCam, np.mean(self.cams[thisCam - 1].wavelengths[0])    
                                                   
                    elif thisCam == 4:
                        self.cam4_fluxes.append(HDUList[0].data[idx][0])
                        self.cam4_wavelengths.append(self.extract_HERMES_wavelength(HDUList[0].header))
                        self.cam4_fileNames.append(thisFile)

            sortOrder = np.argsort(self.JDs)         
            if thisCam == 1:
                self.UTdates = np.array(self.UTdates)[sortOrder]
                self.UTstarts = np.array(self.UTstarts)[sortOrder]
                self.UTends = np.array(self.UTends)[sortOrder]
                self.lengths = np.array(self.lengths)[sortOrder]
                self.JDs = np.array(self.JDs)[sortOrder]
                self.HRs = np.array(self.HRs)[sortOrder]
                self.cam1_fluxes = np.array(self.cam1_fluxes)[sortOrder]
                self.cam1_wavelengths = np.array(self.cam1_wavelengths)[sortOrder]
                self.cam1_fileNames = np.array(self.cam1_fileNames)[sortOrder]
#                 print self.cam1_wavelengths.shape
                self.cam1_RVs.bulk_calculate_RV(self.cam1_wavelengths, self.cam1_fluxes, xDef)

            elif thisCam == 2:
                self.cam2_fluxes = np.array(self.cam2_fluxes)[sortOrder]
                self.cam2_wavelengths = np.array(self.cam2_wavelengths)[sortOrder]
                self.cam2_fileNames = np.array(self.cam2_fileNames)[sortOrder]
    
                self.cam2_RVs.bulk_calculate_RV(self.cam2_wavelengths, self.cam2_fluxes, xDef)
                                           
            elif thisCam == 3:
                self.cam3_fluxes = np.array(self.cam3_fluxes)[sortOrder]
                self.cam3_wavelengths = np.array(self.cam3_wavelengths)[sortOrder]
                self.cam3_fileNames = np.array(self.cam3_fileNames)[sortOrder]
    
                self.cam3_RVs.bulk_calculate_RV(self.cam3_wavelengths, self.cam3_fluxes, xDef)
                                           
            elif thisCam == 4:
                self.cam4_fluxes = np.array(self.cam4_fluxes)[sortOrder]
                self.cam4_wavelengths = np.array(self.cam4_wavelengths)[sortOrder]
                self.cam4_fileNames = np.array(self.cam4_fileNames)[sortOrder]

                self.cam4_RVs.bulk_calculate_RV(self.cam4_wavelengths, self.cam4_fluxes, xDef)
                                                   

#3 sigma elimination
#
#             meanRV = np.mean(self.cams[thisCam - 1].RVs.RVs)
#             sigRV = np.std(self.cams[thisCam - 1].RVs.RVs)
#             errRV = np.abs(self.cams[thisCam - 1].RVs.RVs-meanRV)
            
#             threeSigMap = errRV < 3*sigRV
    
#             self.UTdates = np.array(self.UTdates)[threeSigMap]
#             self.UTstarts = np.array(self.UTstarts)[threeSigMap]
#             self.UTends = np.array(self.UTends)[threeSigMap]
#             self.lengths = np.array(self.lengths)[threeSigMap]
#             self.JDs = np.array(self.JDs)[threeSigMap]
#             self.HRs = np.array(self.HRs)[threeSigMap]
#             self.cams[thisCam - 1].fluxes = np.array(self.cams[thisCam - 1].fluxes)[threeSigMap]
#             self.cams[thisCam - 1].wavelengths = np.array(self.cams[thisCam - 1].wavelengths)[threeSigMap]
#             self.cams[thisCam - 1].fileNames = np.array(self.cams[thisCam - 1].fileNames)[threeSigMap]
#             self.cams[thisCam - 1].RVs.RVs = np.array(self.cams[thisCam - 1].RVs.RVs)[threeSigMap]

            os.chdir('..')
            
    def extract_HERMES_wavelength(self, header):
        
        CRVAL1 = header['CRVAL1'] # / Co-ordinate value of axis 1                    
        CDELT1 = header['CDELT1'] #  / Co-ordinate increment along axis 1             
        CRPIX1 = header['CRPIX1'] #  / Reference pixel along axis 1                   
        
        #Creates an array of offset wavelength from the referece px/wavelength
        Lambda = CRVAL1 - (CRPIX1 - (np.arange(int(CRPIX1)*2)) -1)* CDELT1
    
        return Lambda

    

# <codecell>

class RVs():
    from scipy import constants,signal

    
    def __init__(self):
        self.c = constants.speed_of_light
        self.RVs = []
        
    def bulk_calculate_RV(self, wavelengths, fluxes, xDef = 1, xDef2 = 0):
        self.RVs = []
        for i in range(len(wavelengths)):
            print 'Procesing RV ',i
            self.RVs.append(self.calculate_RV(wavelengths[0],fluxes[0], wavelengths[i],fluxes[i], xDef = xDef, xDef2 = xDef2))

    def bulk_calculate_abs_RV(self, wavelengths, fluxes, refWavelenght, refFlux, xDef = 1, xDef2 = 0):
        self.RVs = []
        for i in range(len(wavelengths)):
            print 'Procesing RV ',i
            self.RVs.append(self.calculate_RV( wavelengths[i],fluxes[i],refWavelenght, refFlux, xDef = xDef, xDef2 = xDef2))
        
        
    def calculate_RV(self, lambda1, flux1, lambda2, flux2, xDef = 1, xDef2 = 0):
        lambda1Clean, flux1Clean = self.clean_flux(flux1, xDef, lambda1)
        if xDef2 == 0: xDef2 = xDef
        lambda2Clean, flux2Clean = self.clean_flux(flux2, xDef2, lambda2)
#         plt.plot(lambda1Clean, flux1Clean)
#         plt.plot(lambda2Clean, flux2Clean)
#         plt.show()
        if ((lambda1Clean.shape[0]>0) &
            (flux1Clean.shape[0]>0) &
            (lambda2Clean.shape[0]>0) &
            (flux2Clean.shape[0]>0)):
            a = signal.fftconvolve(flux1Clean, flux2Clean[::-1], mode='same') 
            corrPeak = np.where(a==max(a))[0][0]
            print np.where(a==max(a))
            
#             plt.plot(a)
#             plt.show()
            centralLambda = lambda2Clean[len(lambda2Clean)/2]
            if corrPeak<len(lambda2Clean):
                pass
                shiftLambda = lambda2Clean[corrPeak] - centralLambda
            else:
                pass
                shiftLambda = lambda1Clean[corrPeak] - centralLambda
                
            RV = self.c/centralLambda * shiftLambda
        else:
            RV = 0

        print 'RV', RV
        return RV
    
    
    
    
    def clean_flux(self,flux, xDef = 1, lambdas = np.array([])):
        '''clean a flux array to cross corralate to determine RV shift
            eliminates NaNs
            moving median to reduce peaks
            optional: increase resolution by xDef times
            
        '''	
        
        #Copy to output in case of no resampling
        fluxHD = flux
        newLambdas = lambdas
        
        #if enought data -> resample
        if ((xDef>1) and (len(lambdas)>0)):
            fFluxHD = interpolate.interp1d(lambdas,flux) 
            newLambdas = np.arange(min(lambdas), max(lambdas),(max(lambdas)-min(lambdas))/len(lambdas)/xDef)
            fluxHD = fFluxHD(newLambdas)
        
        #clean NaNs and median outliers	
        fluxHD[np.isnan(fluxHD)] = 0
        fluxNeat = fluxHD	
        fluxMed = signal.medfilt(fluxHD,5)
        w = np.where(abs((fluxHD-fluxMed)/np.maximum(fluxMed,50)) > 0.4)
        for ix in w[0]:
            fluxNeat[ix] = fluxMed[ix]
        
        #remove trailing zeros, devide by fitted curve (flatten) and apply tukey window
        fluxNeat = np.trim_zeros(fluxNeat,'f') 
        newLambdas = newLambdas[-len(fluxNeat):]
        fluxNeat = np.trim_zeros(fluxNeat,'b') 
        newLambdas = newLambdas[:len(fluxNeat)]
        
        if ((newLambdas.shape[0]>0) &  (fluxNeat.shape[0]>0)):
            fFluxNeat = optimize.curve_fit(self.cubic, newLambdas, fluxNeat, p0 = [1,1,1,1])
            fittedCurve = self.cubic(newLambdas, fFluxNeat[0][0], fFluxNeat[0][1], fFluxNeat[0][2], fFluxNeat[0][3])
        # 	plt.plot(fittedCurve)
        # 	plt.plot(fluxNeat)
        # 	plt.show()
        # 	plt.plot(fluxNeat/fittedCurve-1)
        # 	plt.show()
            
            fluxFlat = fluxNeat/fittedCurve-1
            
            fluxWindow = fluxFlat * self.tukey(0.1, len(fluxFlat))
        else:
            fluxWindow = np.array([])
            print 'empty after removing zeros'
            
        return newLambdas, fluxWindow

    def cubic(self,x,a,b,c,d):
        return a*x**3+b*x**2+c*x+d

    
    def tukey(self, 
              alpha, N):

        tukey = np.zeros(N)
        for i in range(int(alpha*(N-1)/2)):
            tukey[i] = 0.5*(1+np.cos(np.pi*(2*i/alpha/(N-1)-1)))
        for i in range(int(alpha*(N-1)/2),int((N-1)*(1-alpha/2))):
            tukey[i] = 1
        for i in range(int((N-1)*(1-alpha/2)),int((N-1))):
            tukey[i] = 0.5*(1+np.cos(np.pi*(2*i/alpha/(N-1)-2/alpha+1)))
        
        return tukey

def colect_all_targets():
    #creates a list of all target names in the current folder
     
    files = glob.glob('*.fits')   
    rawTargetList = np.array([])
     
    for thisFile in files:    
        HDUList = pf.open(thisFile)
        rawTargetList = np.append(rawTargetList,np.asarray(HDUList['FIBRES'].data['NAME']))
     
#     mask = ((rawTargetList!='PARKED') & (rawTargetList!='FIBRE NOT IN USE') & (rawTargetList.find('S', 0, 1)!=0) & (rawTargetList.find('5', 0, 1)!=0))
    mask = ((rawTargetList!='PARKED') & (rawTargetList!='FIBRE NOT IN USE'))
 
#     print np.sum(mask), mask.shape
    return np.unique(rawTargetList[mask])


def get_wavelength(start_wave, wave_per_pixel, size):
    """
    Obtain an array of wavelengths according to input values.

    Parameters
    ----------

    start_wave: float,
        Starting wavelength.

    wave_per_pixel: float,
        Wavelength per pixel.

    size: int,
        Size of array.

    Returns
    -------

    wave_array: numpy.ndarray,
        Wavelength array
    """
    
    return np.array([start_wave + i*wave_per_pixel for i in range(size)])


def get_wstart(ref, wave_ref, wave_per_pixel):
    """
    Obtain the starting wavelength of a spectrum.

    Parameters
    ----------

    ref: int,
        Reference pixel.
    
    wave_ref: float,
        Coordinate at reference pixel.

    wave_per_pixel: float,
        Coordinate increase per pixel.

    Returns
    -------

    wstart: float,
        Starting wavelength.
    """
    
    return wave_ref - ((ref-1) * wave_per_pixel)


def get_wavelength(start_wave, wave_per_pixel, size):
    """
    Obtain an array of wavelengths according to input values.

    Parameters
    ----------

    start_wave: float,
        Starting wavelength.

    wave_per_pixel: float,
        Wavelength per pixel.

    size: int,
        Size of array.

    Returns
    -------

    wave_array: numpy.ndarray,
        Wavelength array
    """
    
    return np.array([start_wave + i*wave_per_pixel for i in range(size)])

def load_spectrum(fname):
    """
    Loads the spectrum in FITS format to a numpy.darray.

    Parameters
    ----------

    fname: str,
        File name of the FITS spectrum.

    Returns
    -------

    spectrum: ndarray,
        Spectrum array with wavelength and flux.
    """
    
    # Load spectrum
    spec_FITS = pf.open(fname)
    #Load flux
    flux = spec_FITS[0].data[0]
    
    #Obtain parameters for wavelength determination from header
    ref_pixel = spec_FITS[0].header['CRPIX1']       #Reference pixel
    coord_ref_pixel = spec_FITS[0].header['CRVAL1'] #Wavelength at reference pixel
    wave_pixel = spec_FITS[0].header['CDELT1']      #Wavelength per pixel
    
    #Get starting wavelength
    wstart = get_wstart(ref_pixel, coord_ref_pixel, wave_pixel)
    
    #Obtain array of wavelength
    wave = get_wavelength(wstart, wave_pixel, len(flux))
#     print wave
    return np.dstack((wave, flux))[0]
