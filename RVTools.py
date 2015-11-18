
# coding: utf-8

# In[5]:

import numpy as np
import pylab as plt
from scipy import interpolate, signal, optimize, constants, stats
from scipy.optimize import leastsq
import pyfits as pf
import sys
import os
from lmfit import minimize, Parameters


# In[ ]:

# sys.path = ['', '/disks/ceres/makemake/aphot/kalumbe/reductions/NGC2477_1arc_6.2','/usr/local/yt-hg', '/home/science/staff/kalumbe/my-astro-lib', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/usr/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages/PILcompat', '/usr/lib/python2.7/dist-packages/gtk-2.0', '/usr/lib/pymodules/python2.7', '/usr/lib/python2.7/dist-packages/ubuntu-sso-client', '/usr/lib/python2.7/dist-packages/ubuntuone-client', '/usr/lib/python2.7/dist-packages/ubuntuone-couch', '/usr/lib/python2.7/dist-packages/ubuntuone-storage-protocol', '/usr/lib/python2.7/dist-packages/wx-2.8-gtk2-unicode']


# In[7]:

def fit_sine_RVs(baryRVs, MJDs,data, RVClip = 1e6, starIdx = -1, cam = -1, npyName = 'sineFit.npy'):

    sine_fit = np.ones((baryRVs.shape[0],4,4))*np.nan
    try: 
        sine_fit = np.load('npy/'+npyName)
        print 'Found previous',npyName,'array. Using it for update'
    except:
        pass
    
    for cam in range(4):
        for i,thisRVs in enumerate(baryRVs[:,:,cam]):
            if ((starIdx==-1) or(starIdx==i)): 
                thisRVs[np.abs(np.nan_to_num(thisRVs))>RVClip]=np.nan
                thisBaryRVs = thisRVs[-np.isnan(thisRVs)]
                thisMJDs = MJDs[-np.isnan(thisRVs)]
                
                print 'This target:', data[i,0], '- Camera ',str(cam+1)
                print 'Calculating RV fit for', thisBaryRVs.shape[0], 'data points.',np.sum(np.isnan(thisRVs)),'NaNs'
                if thisMJDs.shape[0]>3:                
                    minIdx = np.where(thisBaryRVs==np.min(thisBaryRVs))[0][0]
                    maxIdx = np.where(thisBaryRVs==np.max(thisBaryRVs))[0][0]
                    guess_P = np.abs(thisMJDs[minIdx] - thisMJDs[maxIdx])*2

                    params = Parameters()
                    params.add('amp', value=np.abs((np.max(thisBaryRVs)-np.min(thisBaryRVs))/2.), min=0)
                    params.add('phase', value = 0, max = guess_P/2., min = -guess_P/2. )
                    params.add('period', value=guess_P, min = 0 )

                    print 'Guess A,ph,P',params.values()

                    output = minimize(optimise_sine, params, args=(thisMJDs, thisBaryRVs))
                    
                    print 'std err',output.chisqr, output.redchi , np.std(output.residual)

                    sine_fit[i,0,cam]=output.params.valuesdict()['amp']
                    sine_fit[i,1,cam]=output.params.valuesdict()['phase']
                    sine_fit[i,2,cam]=output.params.valuesdict()['period']
                    sine_fit[i,3,cam]=np.std(output.residual)

                    print 'Guess A,ph,P', sine_fit[i,:3,cam]

                else: 
                    print 'Result','NOT ENOUGH DATAPOINTS'

    #                 est_std, est_phase, est_P =ouput[0]

                    sine_fit[i,:,cam]=[0,0,0,0]
                print 
            
    np.save('npy/'+npyName,sine_fit)
    print 'Fine'


# In[3]:

def avg_MJDs_groups(MJDs, gapMins = 40):
    
    mins2Days = 1/24./60
    gap2Days = mins2Days * gapMins # diff in days between exposures to be considered different exposures
    
    diffArray = MJDs.copy()
    diffArray[0] = 0
    diffArray[1:] = MJDs[1:]-MJDs[:-1]
    
    avgIdxGroups = diffArray.copy()
    
    avgIdx = 0
    for i,diff in enumerate(diffArray):
        if diff>gap2Days: avgIdx+=1
        avgIdxGroups[i] = avgIdx
        
    np.save('npy/avgIdxGroups.npy',avgIdxGroups.astype(int))


# In[ ]:

def avg_MJDs_data():

    avgIdxGroups = np.load('npy/avgIdxGroups.npy')
    baryRVs = np.load('npy/baryRVs.npy')
    MJDs = np.load('npy/MJDs.npy')
    
    ttlGroups = np.max(avgIdxGroups)+1
    avgBaryRVs = np.zeros((baryRVs.shape[0],ttlGroups,4))
    avgMJDs = np.zeros(ttlGroups)
    
    for i in range(ttlGroups):
        print i,np.nanmean(baryRVs[:,avgIdxGroups==i,:],axis=1)
        avgBaryRVs[:,i,:] = np.nanmean(baryRVs[:,avgIdxGroups==i,:],axis=1)
        avgMJDs[i] = np.nanmean(MJDs[avgIdxGroups==i])
            
    np.save('npy/avgBaryRVs.npy',avgBaryRVs)
    np.save('npy/avgMJDs.npy',avgMJDs)
    print 'Fine'


# In[ ]:

def optimise_sine(x, MJDs, thisBaryRVs):
    
#     print 'x',x
    
    amp = x['amp'].value
    ph = x['phase'].value
    period = x['period'].value

#     print 'MJDs',MJDs
#     print 'thisBaryRVs',thisBaryRVs.shape
#     if x[2]==0: x[2]=1e-17
#     result = x[0]*(np.sin(np.pi*2./x[2]*(MJDs+x[1]))-np.sin(np.pi*2./x[2]*(MJDs[0]+x[1]))) - thisBaryRVs
#     if x[2]==0: x[2]=1e-17
    #subtracts epoch 0
#     result = amp*(np.sin(np.pi*2./period*(MJDs+ph))- np.sin(np.pi*2./period*(MJDs[0]+ph))) - thisBaryRVs

    #direct fit
    result = amp*(np.sin(np.pi*2./period*(MJDs+ph))) - thisBaryRVs
#     print 'result',result
#     print result - thisBaryRVs
    
    return result 



# In[ ]:

def fit_results(data, sineFit, avgSineFit, baryRVs, avgBaryRVs):

    Range = np.ptp(np.nan_to_num(baryRVs),axis=1)
    order = np.argsort(Range[:,0])[::-1]    
    fittedRangeResults = Range[order]

    ratios = sineFit[:,0,:]/sineFit[:,3,:]
    fittedResults = ratios[order]
    
    
    avgRatios = avgSineFit[:,0,:]/avgSineFit[:,3,:]
    fittedAvgResults = avgRatios[order]
    
    AvgRange = np.ptp(np.nan_to_num(avgBaryRVs),axis=1)
    fittedAvgRangeResults = AvgRange[order]
    
    fittedData = data[:,0][order]
    
    np.save('npy/fittedResults.npy',fittedResults)
    np.save('npy/fittedRangeResults.npy',fittedRangeResults)
    np.save('npy/fittedAvgResults.npy',fittedAvgResults)
    np.save('npy/fittedAvgRangeResults.npy',fittedAvgRangeResults)
    np.save('npy/fittedData.npy',fittedData)
    print 'Fine'


# In[1]:

def check_equal_wl(wl1, wl2):
    result = False
    
    diff = np.sum(wl1-wl2)
    
    if diff<1e-17:
        result = True
    return result


# In[1]:

def pivot2idx(pivot):
    pivot = np.array(pivot)
    
    rev_num = [np.nan] + ((np.tile(np.arange(10,0,-1),40)+np.repeat(np.arange(0,40)*10,10))-1).tolist()
    rev_num = np.array(rev_num)
    
    return rev_num[pivot]


# In[ ]:

def idx2pivot(idx):

    rev_num = [np.nan] + ((np.tile(np.arange(10,0,-1),40)+np.repeat(np.arange(0,40)*10,10))-1).tolist()
    
    if len(np.where(rev_num==idx))>0:
        result = np.where(rev_num==idx)[0][0]
    else:
        result = np.nan

    return result


# In[2]:

def resample_sp(wavelength, flux, minWL, maxWL, xStep):
    '''
    Rebins the flux into xSteps in log_e space
    
    Parameters
    ----
    wavelength : 1-D array
            Wavelength values 
            
    flux : 1-D array
        Flux values 
        
    xStep : float
        Increase step between pixels in log_e(wl)
        
        
    Returns
    ------
    new_wavelength: 
        log_e(wl) in xStep steps
    
    new_flux: 
        rebinned flux

    '''
    
    lnWavelength = np.log(wavelength)
    fFlux = interpolate.splrep(lnWavelength, flux) 
    new_wavelength = np.arange(np.log(minWL), np.log(maxWL),xStep)
    new_flux = interpolate.splev(new_wavelength, fFlux, der=0)
    
#     plt.plot(np.log(wavelength),flux)
#     plt.plot(new_wavelength,new_flux)
#     plt.show()
    
    return new_wavelength, new_flux


# In[20]:

def cubic(x,a,b,c,d):
    '''
    Cubic function
    '''
    return a*x**3+b*x**2+c*x+d


# In[1]:

#Find the common range of wl for all cameras
def find_max_wl_range(thisCam):
    '''
    Checks the range of valid wavelength for all exposures in given camera.
    '''
    if len(thisCam.wavelengths)>0:
        minWL = np.max(np.min(thisCam.wavelengths, axis=1))
        maxWl = np.min(np.max(thisCam.wavelengths, axis=1))

    else:
        minWL, maxWl = 0,0
        
    return minWL, maxWl


# In[ ]:

def clean_arc(wavelength, flux, minWL=0, maxWL=0, xStep = 5*10**-6):
    '''
    Clean a 1D spectrum. 
    
    Parameters
    ----
    wavelength : int or None, optional
        Array of wavelengths 
        
    flux : int or None, optional
        Array of fluxes 
        
    minWL : int, optional
        Minimum walength value to return 
        
    maxWL : int, optional
        Maximum wavelength value to return 
        
    xStep : float, optional
        Coeficient to resample. Final array will be flux.shape[0]*xDef long. 
        
    medianRange : int, optional
        Number of pixels to median over. 0 will skip this step. Optional.

    flatten : boolean, optional
        Divides flux by a fitted 3rd ord polynomial if True. Optional.
        
    Returns
    ----
    wavelength : numpy, floats
        Array of wavelengths 
        
    flux : numpy, floats
        Array of fluxes 
        

    '''
    
    #fix initial nans on edges
    nanMap = np.isnan(flux)
    leftEdgeIdx=0
    rightEdgeIdx=len(flux)
    
#     nanMapIdx = np.where(nanMap==True) <<<<<make the next lines faster by using this
    if np.sum(nanMap)>0:
        print 'Found NaNs in flux array'
    
    for i,booI in enumerate(nanMap):
        if booI==False:
            leftEdgeIdx = i
            break
            
    for j,rbooI in enumerate(nanMap[::-1]):
        if rbooI==False:
            rightEdgeIdx = len(nanMap)-j
            break        

    fluxMedian = stats.nanmedian(flux)
    if leftEdgeIdx>0:
        flux[:leftEdgeIdx] = np.linspace(0, flux[leftEdgeIdx+1],leftEdgeIdx)
    if rightEdgeIdx<len(flux):
        flux[rightEdgeIdx:] = np.linspace(flux[rightEdgeIdx-1], 0, len(flux)-rightEdgeIdx)

        
    if ((wavelength[-np.isnan(flux)].shape[0]>0) &  (flux[-np.isnan(flux)].shape[0]>0)):
        #resample
        wavelength,flux = resample_sp(wavelength, flux, minWL, maxWL, xStep)
        
    else: #if not enough data return NaNs
        wavelength = np.ones(4096)*np.nan
        flux = np.ones(4096)*np.nan
        
    return wavelength, flux



# In[63]:

def clean_flux(wavelength, flux, minWL=0, maxWL=0, xStep = 10**-5, medianRange = 0, flatten = True):
    '''
    Clean a 1D spectrum. 
    
    Parameters
    ----
    wavelength : int or None, optional
        Array of wavelengths 
        
    flux : int or None, optional
        Array of fluxes 
        
    minWL : int, optional
        Minimum walength value to return 
        
    maxWL : int, optional
        Maximum wavelength value to return 
        
    xStep : float, optional
        Coeficient to resample. Final array will be flux.shape[0]*xDef long. 
        
    medianRange : int, optional
        Number of pixels to median over. 0 will skip this step. Optional.

    flatten : boolean, optional
        Divides flux by a fitted 3rd ord polynomial if True. Optional.
        
    Returns
    ----
    wavelength : numpy, floats
        Array of wavelengths 
        
    flux : numpy, floats
        Array of fluxes 
        

    '''
    
    #fix initial nans on edges
    nanMap = np.isnan(flux)
    leftEdgeIdx=0
    rightEdgeIdx=len(flux)
    
#     nanMapIdx = np.where(nanMap==True) <<<<<make the next lines faster by using this
    if np.sum(nanMap)>0:
        print 'Found NaNs in flux array'
        
    for i,booI in enumerate(nanMap):
        if booI==False:
            leftEdgeIdx = i
            break
            
    for j,rbooI in enumerate(nanMap[::-1]):
        if rbooI==False:
            rightEdgeIdx = len(nanMap)-j
            break        

    fluxMedian = stats.nanmedian(flux)
    if leftEdgeIdx>0:
        flux[:leftEdgeIdx] = np.linspace(fluxMedian, flux[leftEdgeIdx+1],leftEdgeIdx)
    if rightEdgeIdx<len(flux):
        flux[rightEdgeIdx:] = np.linspace(flux[rightEdgeIdx-1], fluxMedian, len(flux)-rightEdgeIdx)

        
        
    #median outliers
    if medianRange>0:
        fluxMed = signal.medfilt(flux,medianRange)
        fluxDiff = abs(flux-fluxMed)
#         fluxDiff = flux-fluxMed
        fluxDiffStd = np.std(fluxDiff)
        mask = fluxDiff> 3 * fluxDiffStd
        flux[mask] = fluxMed[mask]


    if ((wavelength[-np.isnan(flux)].shape[0]>0) &  (flux[-np.isnan(flux)].shape[0]>0)):
        
        if flatten==True:#flatten curve by fitting a 3rd order poly
            fFlux = optimize.curve_fit(cubic, wavelength[-np.isnan(flux)], flux[-np.isnan(flux)], p0 = [1,1,1,1])
            fittedCurve = cubic(wavelength, fFlux[0][0], fFlux[0][1], fFlux[0][2], fFlux[0][3])
            flux = flux/fittedCurve-1
        else:
            flux = flux/fluxMedian-1
            
        #apply tukey
        flux = flux * signal.tukey(len(flux), 0.2)

        #resample
        wavelength,flux = resample_sp(wavelength, flux, minWL, maxWL, xStep)
        
    else: #if not enough data return NaNs
        wavelength = np.ones(4096)*np.nan
        flux = np.ones(4096)*np.nan
        
    return wavelength, flux



# In[ ]:

#Create cross correlation curves wrt epoch 0
def RVs_CC_t0_arc(thisStar, CCReferenceSet = 0, corrHWidth=10):
    '''
    Cross-correlates all epochs wrt t0. Writes the results to thisStar.cameras[].RVs
    
    
    Parameters
    ----
    thisStar : obj
        Object holding the star that holds the fluxes to be cross-correlated
        
    starIdx : int
        Index of the star in to be linked in the comments. Not very useful. 
        
        
    Returns
    ------
    Nottin.
    
    '''
    

#     validDates = np.all([np.nansum(thisCam.red_fluxes,1).astype(bool) for thisCam in thisStar.exposures.cameras],0)
    print ''
    for cam,thisCam in enumerate(thisStar.exposures.cameras[:1]):
        RVs = []
        sigmas = [] 
        Qs = []
        SNRs = []

        validDates = np.nansum(thisCam.red_fluxes,1).astype(bool)
        
        print 'Camera',cam
        
        minWL, maxWL = find_max_wl_range(thisCam)
        
        #Filters exposures to a minimum MJD
#         if minMJD>0:
#             print 'Reducing MJD to >=',minMJD
#             validDates = thisStar.exposures.MJDs>=minMJD
            
            
#         if len(np.arange(len(validDates))[validDates])>0:
#             CCReferenceSet = np.arange(len(validDates))[validDates][0]
#         else:
#             CCReferenceSet = 0
            
#         print 'Refernce set =',CCReferenceSet
        
        lambda1, flux1 = clean_arc(thisCam.wavelengths[CCReferenceSet], thisCam.red_fluxes[CCReferenceSet], minWL, maxWL)
        
        plts = 0    
        for epoch, MJD in enumerate(thisStar.exposures.MJDs):
            print epoch,
            lambda2, flux2 = clean_arc(thisCam.wavelengths[epoch], thisCam.red_fluxes[epoch], minWL, maxWL)

            try:
                
                #Duncan's approach to CC. 
                CCCurve = np.correlate(flux1, flux2, mode='full')

                y = CCCurve[int(CCCurve.shape[0]/2.)-corrHWidth:int(CCCurve.shape[0]/2.)+1+corrHWidth].copy()
                y /=np.max(y)
                x = np.arange(-corrHWidth,corrHWidth+1)
                p,_ = fit_flexi_gaussian([1,3.,2.,1.,0],y,x )
                shift = p[0]
                
#                 thisQ, thisdRV = QdRV(thisCam.wavelengths[epoch], thisCam.red_fluxes[epoch])

                px = 1000
                RV = (np.exp(lambda1[px+1]-lambda1[px]) -1) * constants.c * shift
                print 'RV',RV                

            except Exception,e: 
                print 'CC Error'
                print str(e)
                R = 0
                thisQ = 0
                thisdRV = 0
                RV = 0

            SNR = np.sqrt(stats.nanmedian(thisCam.red_fluxes[epoch]))


            SNRs.append(SNR)
    #         Qs.append(thisQ)
    #         sigmas.append(thisdRV)
            RVs.append(RV)



#     thisCam.sigmas = np.array(sigmas)
#     thisCam.Qs = np.array(Qs)
    thisCam.RVs = np.array(RVs)
    thisCam.SNRs = np.array(SNRs)


# In[ ]:

def gaussian(x, mu, sig, ):
    x = np.array(x)
    return np.exp(-np.power(x - mu, 2.) / 2 / np.power(sig, 2.))


def flexi_gaussian(x, mu, sig, power, a, d ):
    x = np.array(x)
    return a* np.exp(-np.power(np.abs((x - mu) * np.sqrt(2*np.log(2))/sig),power))+d

def fit_gaussian(p, flux, x_range):
    a = optimize.leastsq(diff_gaussian, p, args= [flux, x_range])
    return a

def fit_flexi_gaussian(p, flux, x_range):
    a = optimize.leastsq(diff_flexi_gaussian, p, args= [flux, x_range])
    return a

def diff_gaussian(p, args):
    
    flux = args[0]
    x_range = args[1]

    diff = gaussian(x_range, p[0],p[1]) - flux
    return diff

def diff_flexi_gaussian(p, args):
    
    flux = args[0]
    x_range = args[1]
    weights = np.abs(np.gradient(flux)) * (flux+np.max(flux)*.1)
    diff = (flexi_gaussian(x_range, p[0], p[1], p[2], p[3], p[4]) - flux)# *weights
    return diff


# In[2]:

#Create cross correlation curves wrt epoch 0
def RVs_CC_t0(thisStar, starIdx, minMJD=0 ,  xDef = 1, CCReferenceSet = 0, printDetails=False, corrHWidth=10, medianRange = 0, useRangeFilter = False):
    '''
    Cross-correlates all epochs wrt t0. Writes the results to thisStar.cameras[].RVs
    
    
    Parameters
    ----
    thisStar : obj
        Object holding the star that holds the fluxes to be cross-correlated
        
    starIdx : int
        Index of the star in to be linked in the comments. Not very useful. 
        
        
    Returns
    ------
    Nottin.
    
    '''
    

#     validDates = np.all([np.nansum(thisCam.red_fluxes,1).astype(bool) for thisCam in thisStar.exposures.cameras],0)
    print ''
    for cam,thisCam in enumerate(thisStar.exposures.cameras[:1]):
        RVs = []
        sigmas = [] 
        Qs = []
        SNRs = []

        validDates = np.nansum(thisCam.red_fluxes,1).astype(bool)
        
        print 'Camera',cam
        
        minWL, maxWL = find_max_wl_range(thisCam)
        
        #Filters exposures to a minimum MJD
#         if minMJD>0:
#             print 'Reducing MJD to >=',minMJD
#             validDates = thisStar.exposures.MJDs>=minMJD
            
            
#         if len(np.arange(len(validDates))[validDates])>0:
#             CCReferenceSet = np.arange(len(validDates))[validDates][0]
#         else:
#             CCReferenceSet = 0
            
#         print 'Refernce set =',CCReferenceSet
        
        lambda1, flux1 = clean_flux(thisCam.wavelengths[CCReferenceSet], thisCam.red_fluxes[CCReferenceSet], minWL, maxWL, medianRange=medianRange)
        
        plts = 0    
        for epoch, MJD in enumerate(thisStar.exposures.MJDs):
            print epoch,
            lambda2, flux2 = clean_flux(thisCam.wavelengths[epoch], thisCam.red_fluxes[epoch], minWL, maxWL, medianRange=medianRange)

            try:
                
                #Duncan's approach to CC. 
                CCCurve = np.correlate(flux1, flux2, mode='full')

                y = CCCurve[int(CCCurve.shape[0]/2.)+1-5:int(CCCurve.shape[0]/2.)+1+4].copy()
                y /=np.max(y)
                x = np.arange(-4,5)
                p,_ = fit_gaussian([1,3.],y,x )
                shift = p[0]
                
#                 thisQ, thisdRV = QdRV(thisCam.wavelengths[epoch], thisCam.red_fluxes[epoch])

                px = 1000
                RV = (np.exp(lambda1[px+1]-lambda1[px]) -1) * constants.c * shift
                print 'RV',RV                

            except Exception,e: 
                print 'CC Error'
                print str(e)
                R = 0
                thisQ = 0
                thisdRV = 0
                RV = 0

            SNR = np.sqrt(stats.nanmedian(thisCam.red_fluxes[epoch]))


            SNRs.append(SNR)
    #         Qs.append(thisQ)
    #         sigmas.append(thisdRV)
            RVs.append(RV)



#     thisCam.sigmas = np.array(sigmas)
#     thisCam.Qs = np.array(Qs)
    thisCam.RVs = np.array(RVs)
    thisCam.SNRs = np.array(SNRs)


# In[24]:

def comment(star, epoch, cam, comment):
    comments = []
    try:
        os.mkdir('npy')
    except:
        pass
    
    try:
        comments = np.load('npy/comments.npy')
    except:
        pass
    
    if comments==[]:
        comments = np.zeros((1,),dtype=('i4,i4,i4,a100'))
        comments[:] = [(star, epoch, cam, comment)]
    else:
        x = np.zeros((1,),dtype=('i4,i4,i4,a100'))
        x[:] = [(star, epoch, cam, comment)]
        comments = np.append(comments,x)
    
    np.save('npy/comments.npy',comments)
        


# In[25]:

#Create cross correlation curves wrt epoch 0
def single_RVs_CC_t0(thisStar, cam = 0, t = 0, corrHWidth =10, xDef = 1):

        print 'Camera',cam, '- t0 wrt t',t
        
        thisCam = thisStar.exposures.cameras[cam]
            
        lambda1, flux1 = clean_flux(thisCam.wavelengths[0], thisCam.red_fluxes[0], thisCam, medianRange=5, xDef=xDef )
        
        lambda2, flux2 = clean_flux(thisCam.wavelengths[t], thisCam.red_fluxes[t], thisCam, medianRange=5, xDef=xDef)
        CCCurve = []
        CCCurve = signal.fftconvolve(flux1[-np.isnan(flux1)], flux2[-np.isnan(flux2)][::-1], mode='same')
        corrMax = np.where(CCCurve==max(CCCurve))[0][0]
        p_guess = [corrMax,corrHWidth]
        x_mask = np.arange(corrMax-corrHWidth, corrMax+corrHWidth+1)
        if max(x_mask)<len(CCCurve):
            p = fit_gaussian(p_guess, CCCurve[x_mask], np.arange(len(CCCurve))[x_mask])[0]
            if np.modf(CCCurve.shape[0]/2.0)[0]>1e-5:
                pixelShift = (p[0]-(CCCurve.shape[0]-1)/2.) #odd number of elements
            else:
                pixelShift = (p[0]-(CCCurve.shape[0])/2.) #even number of elements


            mid_px = thisCam.wavelengths.shape[1]/2
            dWl = (thisCam.wavelengths[t,mid_px+1]-thisCam.wavelengths[t,mid_px]) / thisCam.wavelengths[t,mid_px]/xDef
            RV = dWl * pixelShift * constants.c 
            print 'RV',RV
        else:
            p=RV=0
            
#         print 'HERE:'

        return lambda1,flux1, lambda2,flux2, CCCurve, p, x_mask, RV 
        


# In[26]:

#Bouchy functions
def QdRV(Lambda, A0):
	
	W1 = W(Lambda, A0)
	Q_out = 0
	dRV = 0
	if np.sum(W1)>0:
		Q_out = Q(W1, A0)
		dRV = constants.c/np.sqrt(np.sum(W1))
	
	return Q_out, dRV

def Q(W, A0):
	'''
    Calculates the Q factor of a spectrum from W(weight) and A0(flux) form Bouchy 2001.
    
    Parameters
    ----------
    W : np.array
        n x 1 np.array weight 
        
    AO : np.array
        n x 1 np.array with flux counts
    
            
    Returns
    -------
    Q : float
        Quality factor. 
        
    Notes
    -----

    '''
	Q = 0
	if np.sum(A0[-np.isnan(A0)])>0:
		Q = np.sqrt(np.sum(W)/np.sum(A0[-np.isnan(A0)]))
	
	return Q




def W(Lambda, A0):
	'''
    Calculates the weight function form Bouchy 2001.
    
    Parameters
    ----------
    Lambda : np.array
        n x 1 np.array with wavelength bins
        
    AO : np.array
        n x 1 np.array with counts
    
            
    Returns
    -------
    W : np.array
        n x 1 np.array weights as a function of pixel.
        
    Notes
    -----
    Lambda and A0 should be equal length.
    Uses:
    W(i) = Lambda(i)**2 (dA0(i)/dLambda(i))**2 / A0(i)
    Assumes noise free detector. (No sigma_D**2 term in the denominator).
    dA0(i)/dLambda(i) simplified as discrete DeltaY/DeltaX.
    '''

	dA0dL = np.zeros(len(A0)-1)
	
	for i in range(len(A0)-1): #compute partial derivative
		dA0dL[i] = (A0[i+1] - A0[i])/(Lambda[i+1] - Lambda[i])

	#compute W (removing last term from Lambda and A0 as dA0dL has n-1 terms.
	W = Lambda[:-1]**2 * dA0dL**2 / A0[:-1]
	
	#clean nans
	W[np.isnan(W)] = 0
	
	return W


# In[27]:

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
    Lambda = CRVAL1 - (CRPIX1 - (np.arange(int(CRPIX1)*2)))* CDELT1

    return Lambda


# In[ ]:

def extract_iraf_wavelength(header, app):
    WS = 'WS_'+str(app)
    WD = 'WD_'+str(app)

    first_px = float(header[WS])
    disp = float(header[WD])
    length = header['NAXIS1']
    wl = np.arange(length)*disp
    wl += first_px
    
    return wl


# In[ ]:

def extract_pyhermes_wavelength(fileName):

    thisFile = pf.open(fileName)

    CRVAL1 = thisFile[0].header['CRVAL1'] # / Co-ordinate value of axis 1                    
    CDELT1 = thisFile[0].header['CDELT1'] #  / Co-ordinate increment along axis 1             
    CRPIX1 = thisFile[0].header['CRPIX1'] #  / Reference pixel along axis 1                   
    NAXIS1 = thisFile[0].header['NAXIS1'] #  / length of the array     

    #Creates an array of offset wavelength from the referece px/wavelength
    Lambda = (np.arange(int(NAXIS1)))* CDELT1 + CRVAL1
    
    return Lambda


# In[28]:

def pivot_to_y(ref_file):
     
    a = pf.getdata(ref_file)
    
    return a[:,200]


# In[29]:


def calibrator_weights(deltay, sigma):
    """For calibrator stars with CCD y values deltay from the target star
    and radial velocity errors sigma, create an optimal set of weights.

    We want to minimise the variance of the weighted sum of calibrator
    radial velocities where we have the following constraints:

    1) \Sigma w_i = 1  (i.e. the average value of the calibrators measure CCD shifts)
    2) \Sigma w_i dy_i = 0 (i.e. allow the wavelength solution to rotate about the target)

    See http://en.wikipedia.org/wiki/Quadratic_programming
    """
    N = len(sigma)
    #Start of with a matrix of zeros then fill it with the "Q" and "E" matrices
    M = np.zeros((N+2,N+2))
    M[(range(N),range(N))] = sigma
#     idx = np.where(deltay==0)[0][0]
#     M[idx,idx] = 1e17
    M[N,0:N] = deltay
    M[0:N,N] = deltay
    M[N+1,0:N] = np.ones(N)
    M[0:N,N+1] = np.ones(N)
    b = np.zeros(N+2)
    b[N+1] = 1.0
    #Solve the problem M * x = b
    x = np.linalg.solve(M,b)
    #The first N elements of x contain the weights.
    return x[0:N]


# In[30]:

def calibrator_weights2(deltay,SNR):

    c = 1/np.abs(deltay)/SNR
    c[deltay==0]=0
    c /=np.sum(c)
    return c


# In[31]:

def calibrator_weights3(deltay,SNR):
#nope
    c = (SNR+np.abs(deltay))/np.abs(deltay)
    c[deltay==0]=0
    c /=np.sum(c)
    return c


# In[32]:

def create_allW(data = [], SNRs = [], starSet=[], RVCorrMethod = 'PM', refEpoch = 0):

    if ((data!=[]) and (SNRs!=[])):
        if ((starSet!=[]) and (len(starSet.shape)==1) and (starSet[0]>0)):
            data = data[starSet]
            SNRs = SNRs[starSet]

        #load function that translates pivot# to y-pixel  p2y(pivot)=y-pixel of pivot
        p2y = pivot_to_y('/Users/Carlos/Documents/HERMES/reductions/6.2/rhoTuc_6.2/0_20aug/1/20aug10042tlm.fits') 

        #gets the y position of for the data array
        datay = p2y[data[:,2].astype(float).astype(int)]
        order = np.argsort(datay)
        
        #Creates empty array for relative weights
        #allW[Weights, camera, staridx of the star to be corrected]
        allW = np.zeros((data.shape[0],4,data.shape[0]))

        for thisStarIdx in range(data.shape[0]):

            #converts datay into deltay
            deltay = datay-datay[thisStarIdx]

            for cam in range(4):

                thisSigma = 1./SNRs[:,refEpoch,cam].copy()
                thisSigma[np.isnan(thisSigma)]=1e+17  #sets NaNs into SNR=1e-17
                
                if np.sum(thisSigma)>0:
                    if RVCorrMethod == 'PM':
                        W = calibrator_weights(deltay,thisSigma)
                    elif RVCorrMethod == 'DM':
                        W = calibrator_weights2(deltay,thisSigma)
                        
#                         print data[thisStarIdx,0],RVCorrMethod
                        if data[thisStarIdx,0]=='Giant01':
                            for a,b,c in zip(thisSigma[order],W[order], thisSigma[order]):
                                print 1./a,b,c
                        print ''

                else:
                    W = np.zeros(deltay.shape[0]) #hack to fix an all zeros SNRs for failed reductions
                
                allW[:,cam,thisStarIdx] = W
                    
    else:
        print 'Create allW: Input arrays missing'
        allW =[]

    return allW


# In[33]:

def create_RVCorr_PM(RVs, allW, RVClip = 1e17,starSet=[]):
    RVCorr = np.zeros(RVs.shape)
    print 'Clipping to',RVClip
    RVs[np.abs(RVs)>RVClip]=0
    
    for thisStarIdx in range(RVs.shape[0]):
        for epoch in range(RVs.shape[1]):
            thisRVCorr = (allW[:,:,thisStarIdx]+1)*RVs[:,epoch,:]
            RVCorr[:,epoch,:] = thisRVCorr + RVs[:,epoch,:] 

    return RVCorr


# In[34]:

def create_RVCorr_DM(RVs, allW, RVClip = 1e17,starSet=[]):
    RVCorr = np.zeros(RVs.shape)
    print 'Clipping to',RVClip
    RVs[np.abs(RVs)>RVClip]=0
    
    for thisStarIdx in range(RVs.shape[0]):
        for epoch in range(RVs.shape[1]):
            for cam in range(RVs.shape[2]):
                thisRVCorr = np.nansum(allW[:,cam,thisStarIdx]*RVs[:,epoch,cam])
                RVCorr[thisStarIdx,epoch,cam] = thisRVCorr

    return RVCorr


# In[35]:

def quad(x,a,b,c):
    curve  = a*x**2+b*x+c
    return curve

def fit_quad(p, quadX, quadY):
    a = optimize.leastsq(diff_quad, p, args= [quadX, quadY], epsfcn=0.1)
    return a

def diff_quad(p, args):
    quadX = args[0]
    quadY = args[1]
    diff = quad(quadX, p[0],p[1], p[2]) - quadY
    return diff


# In[21]:

def tukey(alpha, N):
    '''
    Deprecated, use scipy.signal.tukey
    Creates a tukey function
    
    
    Parameters
    ----
    alpha : float
        Fraction of the pixels to fade in/out.
        i.e. alpha=0.1 will use 10% of the pixels to go from 0 to 1. 
        
    N : int
        Totla number of pixels in the array.
        
        
    Returns
    ------

    N-length array of floats from 0 to 1. 
    '''

    
    tukey = np.zeros(N)
    for i in range(int(alpha*(N-1)/2)):
        tukey[i] = 0.5*(1+np.cos(np.pi*(2*i/alpha/(N-1)-1)))
    for i in range(int(alpha*(N-1)/2),int((N-1)*(1-alpha/2))):
        tukey[i] = 1
    for i in range(int((N-1)*(1-alpha/2)),int((N-1))):
        tukey[i] = 0.5*(1+np.cos(np.pi*(2*i/alpha/(N-1)-2/alpha+1)))
    
    return tukey


# In[36]:

def fit_continuum(disp, flux, knot_spacing=200, sigma_clip=(1.0, 0.2),       max_iterations=3, order=3, exclude=None, include=None,       additional_points=None, function='spline', scale=1.0, **kwargs):
    """Fits the continuum for a given `Spectrum1D` spectrum.
    
    Parameters
    ----
    knot_spacing : float or None, optional
        The knot spacing for the continuum spline function in Angstroms. Optional.
        If not provided then the knot spacing will be determined automatically.
    
    sigma_clip : a tuple of two floats, optional
        This is the lower and upper sigma clipping level respectively. Optional.
        
    max_iterations : int, optional
        Maximum number of spline-fitting operations.
        
    order : int, optional
        The order of the spline function to fit.
        
    exclude : list of tuple-types containing floats, optional
        A list of wavelength regions to always exclude when determining the
        continuum. Example:
        
        >> exclude = [
        >>    (3890.0, 4110.0),
        >>    (4310.0, 4340.0)
        >>  ]
        
        In the example above the regions between 3890 A and 4110 A, as well as
        4310 A to 4340 A will always be excluded when determining the continuum
        regions.

    function: only 'spline' or 'poly'

    scale : float
        A scaling factor to apply to the normalised flux levels.
        
    include : list of tuple-types containing floats, optional
        A list of wavelength regions to always include when determining the
        continuum.
        
    Notes
    -----
    (c) Dr. Casey
    """
    
    exclusions = []
    continuum_indices = range(len(flux))

    # Snip left and right
    finite_positive_flux = np.isfinite(flux) * flux > 0

    #print "finite flux", np.any(finite_positive_flux), finite_positive_flux
    #print "where flux", np.where(finite_positive_flux)
    #print "flux is...", flux
    left_index = np.where(finite_positive_flux)[0][0]
    right_index = np.where(finite_positive_flux)[0][-1]

    # See if there are any regions we need to exclude
    if exclude is not None and len(exclude) > 0:
        exclude_indices = []
        
        if isinstance(exclude[0], float) and len(exclude) == 2:
            # Only two floats given, so we only have one region to exclude
            exclude_indices.extend(range(*np.searchsorted(disp, exclude)))
            
        else:
            # Multiple regions provided
            for exclude_region in exclude:
                exclude_indices.extend(range(*np.searchsorted(disp, exclude_region)))
    
        continuum_indices = np.sort(list(set(continuum_indices).difference(np.sort(exclude_indices))))
        
    # See if there are any regions we should always include
    if include is not None and len(include) > 0:
        include_indices = []
        
        if isinstance(include[0], float) and len(include) == 2:
            # Only two floats given, so we can only have one region to include
            include_indices.extend(range(*np.searchsorted(disp, include)))
            
        else:
            # Multiple regions provided
            for include_region in include:
                include_indices.extend(range(*np.searchsorted(disp, include_region)))
    

    # We should exclude non-finite numbers from the fit
    non_finite_indices = np.where(~np.isfinite(flux))[0]
    continuum_indices = np.sort(list(set(continuum_indices).difference(non_finite_indices)))

    # We should also exclude zero or negative flux points from the fit
    zero_flux_indices = np.where(0 >= flux)[0]
    continuum_indices = np.sort(list(set(continuum_indices).difference(zero_flux_indices)))

    original_continuum_indices = continuum_indices.copy()

    if knot_spacing is None or knot_spacing == 0:
        knots = []

    else:
        knot_spacing = abs(knot_spacing)
        
        end_spacing = ((disp[-1] - disp[0]) % knot_spacing) /2.
    
        if knot_spacing/2. > end_spacing: end_spacing += knot_spacing/2.
            
        knots = np.arange(disp[0] + end_spacing, disp[-1] - end_spacing + knot_spacing, knot_spacing)
        if len(knots) > 0 and knots[-1] > disp[continuum_indices][-1]:
            knots = knots[:knots.searchsorted(disp[continuum_indices][-1])]
            
        if len(knots) > 0 and knots[0] < disp[continuum_indices][0]:
            knots = knots[knots.searchsorted(disp[continuum_indices][0]):]

    for iteration in xrange(max_iterations):
        
        splrep_disp = disp[continuum_indices]
        splrep_flux = flux[continuum_indices]

        splrep_weights = np.ones(len(splrep_disp))

        # We need to add in additional points at the last minute here
        if additional_points is not None and len(additional_points) > 0:

            for point, flux, weight in additional_points:

                # Get the index of the fit
                insert_index = int(np.searchsorted(splrep_disp, point))
                
                # Insert the values
                splrep_disp = np.insert(splrep_disp, insert_index, point)
                splrep_flux = np.insert(splrep_flux, insert_index, flux)
                splrep_weights = np.insert(splrep_weights, insert_index, weight)

        if function == 'spline':
            order = 5 if order > 5 else order
            tck = interpolate.splrep(splrep_disp, splrep_flux,
                k=order, task=-1, t=knots, w=splrep_weights)

            continuum = interpolate.splev(disp, tck)

        elif function in ("poly", "polynomial"):
        
            p = poly1d(polyfit(splrep_disp, splrep_flux, order))
            continuum = p(disp)

        else:
            raise ValueError("Unknown function type: only spline or poly available (%s given)" % (function, ))
        
        difference = continuum - flux
        sigma_difference = difference / np.std(difference[np.isfinite(flux)])

        # Clipping
        upper_exclude = np.where(sigma_difference > sigma_clip[1])[0]
        lower_exclude = np.where(sigma_difference < -sigma_clip[0])[0]
        
        exclude_indices = list(upper_exclude)
        exclude_indices.extend(lower_exclude)
        exclude_indices = np.array(exclude_indices)
        
        if len(exclude_indices) is 0: break
        
        exclusions.extend(exclude_indices)
        
        # Before excluding anything, we must check to see if there are regions
        # which we should never exclude
        if include is not None:
            exclude_indices = set(exclude_indices).difference(include_indices)
        
        # Remove regions that have been excluded
        continuum_indices = np.sort(list(set(continuum_indices).difference(exclude_indices)))
    
    # Snip the edges based on exclude regions
    if exclude is not None and len(exclude) > 0:

        # If there are exclusion regions that extend past the left_index/right_index,
        # then we will need to adjust left_index/right_index accordingly

        left_index = np.max([left_index, np.min(original_continuum_indices)])
        right_index = np.min([right_index, np.max(original_continuum_indices)])
        

    # Apply flux scaling
    continuum *= scale
    return disp, flux/continuum

