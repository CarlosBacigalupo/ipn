# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob
import os
import numpy as np
import scipy.constants as const
# import commands
# import pyfits as pf\n",
# import toolbox\n",
from scipy import signal, interpolate, optimize, constants
import pylab as plt
import pickle
from red_tools import *
# import asteroseismology as ast
# import HERMES
# reload(red_tools)\n",
# reload(ast)
# import carlos
import TableBrowser as tb
from IPython.display import display, Math, Latex
import sys

# <headingcell level=2>

# Functions

# <codecell>

def fit_continuum(disp, flux, knot_spacing=200, sigma_clip=(1.0, 0.2), \
      max_iterations=3, order=3, exclude=None, include=None, \
      additional_points=None, function='spline', scale=1.0, **kwargs):
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


# <codecell>

#Clean fluxes
def clean_flux(flux, xDef = 1, lambdas = np.array([])):
    '''clean a flux array to cross corralate to determine RV shift
    '''	
    
    #median outliers	
    fluxMed = signal.medfilt(flux,5)
    w = np.where(abs((flux-fluxMed)/np.maximum(fluxMed,50)) > 0.4)
    for ix in w[0]:
        flux[ix] = fluxMed[ix]

    if ((lambdas[-np.isnan(flux)].shape[0]>0) &  (flux[-np.isnan(flux)].shape[0]>0)):
        fFlux = optimize.curve_fit(cubic, lambdas[-np.isnan(flux)], flux[-np.isnan(flux)], p0 = [1,1,1,1])
        fittedCurve = cubic(lambdas, fFlux[0][0], fFlux[0][1], fFlux[0][2], fFlux[0][3])
        flux = flux/fittedCurve-1
        
        flux = flux * tukey(0.1, len(flux))

        if (xDef>1):
            fFlux = interpolate.interp1d(lambdas,flux) 
            lambdas = np.linspace(min(lambdas), max(lambdas),len(lambdas)*xDef)
            flux = fFlux(lambdas)

    else:
        if (xDef>1):
            lambdas = np.linspace(min(lambdas), max(lambdas),len(lambdas)*xDef)
            flux = np.ones(lambdas.shape[0])*np.nan
        
    return lambdas, flux

def cubic(x,a,b,c,d):
    return a*x**3+b*x**2+c*x+d
    
def tukey(alpha, N):

    tukey = np.zeros(N)
    for i in range(int(alpha*(N-1)/2)):
        tukey[i] = 0.5*(1+np.cos(np.pi*(2*i/alpha/(N-1)-1)))
    for i in range(int(alpha*(N-1)/2),int((N-1)*(1-alpha/2))):
        tukey[i] = 1
    for i in range(int((N-1)*(1-alpha/2)),int((N-1))):
        tukey[i] = 0.5*(1+np.cos(np.pi*(2*i/alpha/(N-1)-2/alpha+1)))
    
    return tukey

# <codecell>

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


# <codecell>

#flux mask used when cross correlating
def create_flux_mask(thisStar):
    flux_mask=[]
    for j in range(4):
        thisCam = thisStar.exposures.cameras[j]
        thisCam.flux_mask = np.ones(len(thisCam.wavelengths[0])).astype(bool)
    # flux_mask[3] = thisCam.wavelengths[3]>7690

# <codecell>

#Clean fluxes
def create_clean_fluxes(thisStar):
    for j in range(4):
        cleanFluxes = []
        cleanWavelengths = []
        thisCam = thisStar.exposures.cameras[j]
        for i in np.arange(thisStar.exposures.JDs.shape[0]):
            print thisStar.exposures.JDs[i],
            thisFlux = thisCam.red_fluxes[i]
            thisLambdas = thisCam.wavelengths[i]
            lambdaClean, fluxClean = clean_flux(thisFlux, xDef, thisLambdas)
            cleanFluxes.append(fluxClean)
            cleanWavelengths.append(lambdaClean)
                
        thisCam.clean_fluxes = np.array(cleanFluxes)
        thisCam.clean_wavelengths = np.array(cleanWavelengths)
        print ''

# <codecell>

#make all fluxes have the same wavelength range and length
def standarise_fluxes(thisStar):
    for j in range(4):
        thisCam = thisStar.exposures.cameras[j]
        
        #first pass to find ranges
        high_min = 0
        low_max = 1e6
        for i in thisCam.clean_wavelengths:
            if np.min(i)>high_min:
                print 'New min', np.min(i)
                high_min = np.min(i)
                
            if np.max(i)<low_max:
                print 'New max', np.max(i)
                low_max = np.max(i)
                
        #second pass, constrain to equal range
        for i in range(thisCam.clean_wavelengths.shape[0]):
            lam = thisCam.clean_wavelengths[i]
            flux = thisCam.clean_fluxes[i]
            mask = np.where((lam>=high_min) & (lam<=low_max))[0]
            print mask
            thisCam.clean_wavelengths[i] = np.nan *np.ones(thisCam.clean_wavelengths[i].shape[0])
            thisCam.clean_wavelengths[i][mask] = lam[mask]
            thisCam.clean_fluxes[i] = np.nan *np.ones(thisCam.clean_fluxes[i].shape[0])
            thisCam.clean_fluxes[i][mask]= flux[mask]

# <codecell>

#creates mean and median fluxes for CC
def create_mean_median_fluxes(thisStar):
    for j in range(4):
        thisCam = thisStar.exposures.cameras[j]
        thisCam.mean_flux = np.nanmean(thisCam.clean_fluxes[thisCam.safe_flag], axis =0)
        thisCam.median_flux = np.median(thisCam.clean_fluxes[thisCam.safe_flag], axis =0)
        thisCam.median_wavelength = thisCam.clean_wavelengths[0]
        thisCam.mean_wavelength = thisCam.clean_wavelengths[0]

# <codecell>

#Debuging CC - cross correlation - create cross correlation curves wrt MEAN
def create_CC_mean(thisStar):
    for j in range(4):
        CCCurves = []
        thisCam = thisStar.exposures.cameras[j]
        for i in range(thisStar.exposures.JDs.shape[0]):
            print thisStar.exposures.JDs[i], 
            flux1 = thisCam.mean_flux
            lamba1 = thisCam.mean_wavelength
            flux2 = thisCam.clean_fluxes[i]
            lamba2 = thisCam.clean_wavelengths[i]
            flux1[np.isnan(flux1)]=0
            flux2[np.isnan(flux2)]=0
            a = signal.fftconvolve(flux1, flux2[::-1], mode='same')
            CCCurves.append(a)
        thisCam.CCCurves = np.array(CCCurves)
        print ''

# <codecell>

#Debuging CC - cross correlation - create cross correlation curves wrt MEAN
def create_CC_median(thisStar):
    for j in range(4):
        CCCurves = []
        thisCam = thisStar.exposures.cameras[j]
        for i in range(thisStar.exposures.JDs.shape[0]):
            print thisStar.exposures.JDs[i], 
            flux1 = thisCam.median_flux
            lamba1 = thisCam.median_wavelength
            flux2 = thisCam.clean_fluxes[i]
            lamba2 = thisCam.clean_wavelengths[i]
            flux1[np.isnan(flux1)]=0
            flux2[np.isnan(flux2)]=0
            a = signal.fftconvolve(flux1, flux2[::-1], mode='same')
            CCCurves.append(a)
        thisCam.CCCurves = np.array(CCCurves)
        print ''

# <codecell>

#Debuging CC - cross correlation - create cross correlation curves wrt epoch 0
def create_CC_t0(thisStar):
    for j in range(4):
        CCCurves = []
        thisCam = thisStar.exposures.cameras[j]
        for i in range(thisStar.exposures.JDs.shape[0]):
            print thisStar.exposures.JDs[i], 
            flux1 = thisCam.clean_fluxes[0]
            lambda1 = thisCam.clean_wavelengths[0]
            flux2 = thisCam.clean_fluxes[i]
            lambda2 = thisCam.clean_wavelengths[i]
            flux1[np.isnan(flux1)]=0
            flux2[np.isnan(flux2)]=0
            a = signal.fftconvolve(flux1, flux2[::-1], mode='same')

            CCCurves.append(a)
        thisCam.CCCurves = np.array(CCCurves)
        print ''

# <codecell>

#Debuging CC - cross correlation - fit gaussian in CCCurves
def fit_CC(thisStar):
    for j in range(4):
        Ps = []
        shiftedWavelength = []
        shiftedPixel = []
        sigmas = [] 
        Qs = []
        thisCam = thisStar.exposures.cameras[j]
        for i in range(thisStar.exposures.JDs.shape[0]):
            print thisStar.exposures.JDs[i],
            thisCCCurve = thisCam.CCCurves[i]
            if np.sum(thisCCCurve)!=0:
                corrMax = np.where(thisCCCurve==max(thisCCCurve))[0][0]
                p_guess = [corrMax,xDef]
                x_mask = np.arange(corrMax-CCMaskWidth, corrMax+CCMaskWidth+1)
                p = fit_gaussian(p_guess, thisCCCurve[x_mask], np.arange(len(thisCCCurve))[x_mask])[0]
                shiftedLambda = get_wavelength(thisCam.clean_wavelengths[i], p[0])
                Ps.append(p)
                shiftedWavelength.append(shiftedLambda)
                shiftedPixel.append(p[0]-thisCCCurve.shape[0]/2)
                thisQ, thisdRV = QdRV(thisCam.wavelengths[i], thisCam.red_fluxes[i])
                Qs.append(thisQ)
                sigmas.append(thisdRV)
            else:
                print 'Flat CC results. Retuning NaNs'
                Ps.append([np.nan,np.nan])
                shiftedWavelength.append(np.nan)
                sigmas.append(np.nan)
                shiftedPixel.append(np.nan)

                
        thisCam.Ps = np.array(Ps)
        thisCam.shfted_wavelengths = np.array(shiftedWavelength)
        thisCam.shifted_pixel = np.array(shiftedPixel)
        thisCam.sigmas = np.array(sigmas)
        thisCam.Qs = np.array(Qs)
        print ''

# <codecell>

#Creates RVs from shifted_wavelength
def create_RVs(thisStar):

    for j in range(4):
#         testRVs = []
        thisCam = thisStar.exposures.cameras[j]
#         for i in range(thisStar.exposures.JDs.shape[0]):
#             print thisStar.exposures.JDs[i],
#             clean_wavelength_median = np.median(thisCam.clean_wavelengths[i])
#             testRVs.append((clean_wavelength_median - thisCam.shfted_wavelengths[i])/clean_wavelength_median*const.c)
#         testRVs.append(((thisCam.clean_wavelengths[0,1]-thisCam.clean_wavelengths[0,0])
#                        /thisCam.clean_wavelengths[0,0]*const.c*
#                        thisCam.shifted_pixel))
        
        thisCam.RVs = np.array((thisCam.clean_wavelengths[0,1]-thisCam.clean_wavelengths[0,0])
                       /thisCam.clean_wavelengths[0,0]*const.c*
                       thisCam.shifted_pixel)
#     print ''

# <codecell>

#Creates delta RVs from shifted_wavelength DDRV
def create_DDRV(thisStar):
    for j in range(4):
        thisCam = thisStar.exposures.cameras[j]    
    #     sig = np.std(thisCam.RVs[thisStar.exposures.my_data_mask])
    #     sigMask = thisCam.RVs[thisStar.exposures.my_data_mask]-np.mean(thisCam.RVs[thisStar.exposures.my_data_mask])<3*sig
        thisCam.DDRVs = thisCam.RVs-np.median(thisCam.RVs[thisCam.safe_flag])

# <codecell>

#Bouchy functions
def QdRV(Lambda, A0):
	
	W1 = W(Lambda, A0)
	Q_out = 0
	dRV = 0
	if np.sum(W1)>0:
		Q_out = Q(W1, A0)
		dRV = const.c/np.sqrt(np.sum(W1))
	
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


# <codecell>


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

# <codecell>

def pivot_to_y(ref_file):
    
    a = pf.getdata(ref_file)
    
    return a[:,200]

# <codecell>

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


# <headingcell level=2>

# Code starts here

# <headingcell level=3>

# Take initial data from [star_name].obj and create global arrays

# <codecell>

os.chdir('/Users/Carlos/Documents/HERMES/reductions/HD1581_single_arc_5.76/combined/')

# <codecell>

a = tb.FibreTable('cam1/20aug10053red.fits')
starNames = a.target[a.type=='P']
print len(starNames)

# <codecell>

starNames[15]

# <codecell>

#takes each <star_name>.obj and calculates RVs, etc. Saves red_<star_name>.obj
xDef = 10 #resampling points per pixel
CCMaskWidth = 5 #half width around the peak of the cc curve to be gaussian fitted
for i in starNames[15:16]:
    print i
#     try:
    filename = i+'.obj'
    filehandler = open(filename, 'r')
    thisStar = pickle.load(filehandler)
#     create_flux_mask(thisStar)
    create_clean_fluxes(thisStar)
    create_mean_median_fluxes(thisStar)
#     create_CC_t0(thisStar)
#     create_CC_mean(thisStar)
    create_CC_median(thisStar)
    fit_CC(thisStar)
    create_RVs(thisStar)
    create_DDRV(thisStar)
    file_pi = open('red_'+thisStar.name+'.obj', 'w') 
    pickle.dump(thisStar, file_pi) 
    file_pi.close()
    thisStar = None
#     except:
#         print 'not created', sys.exc_info()[0]
#         pass

# <codecell>

# Collects information on all stars and writes them into data, RVs, DDRVs, simas, JDs arrays
fileList = glob.glob('/Users/Carlos/Documents/HERMES/reductions/HD1581_single_arc_5.76/combined/red*.obj')

data = []
RVs = np.zeros((len(fileList),4,4))
DDRVs = np.zeros((len(fileList),4,4))
sigmas = np.zeros((len(fileList),4,4))

for i in range(len(fileList)):

    print i,fileList[i]
    filehandler = open(fileList[i], 'r') 
    thisStar = pickle.load(filehandler) 
    data.append([thisStar.name, thisStar.Vmag,np.unique(thisStar.exposures.pivots)[0]])
    RVs[i,:,0] = thisStar.exposures.cameras[0].RVs
    RVs[i,:,1] = thisStar.exposures.cameras[1].RVs
    RVs[i,:,2] = thisStar.exposures.cameras[2].RVs
    RVs[i,:,3] = thisStar.exposures.cameras[3].RVs
    DDRVs[i,:,0] = thisStar.exposures.cameras[0].DDRVs
    DDRVs[i,:,1] = thisStar.exposures.cameras[1].DDRVs
    DDRVs[i,:,2] = thisStar.exposures.cameras[2].DDRVs
    DDRVs[i,:,3] = thisStar.exposures.cameras[3].DDRVs
    sigmas[i,:,0] = thisStar.exposures.cameras[0].sigmas
    sigmas[i,:,1] = thisStar.exposures.cameras[1].sigmas
    sigmas[i,:,2] = thisStar.exposures.cameras[2].sigmas
    sigmas[i,:,3] = thisStar.exposures.cameras[3].sigmas
    JDs = thisStar.exposures.JDs
    filehandler.close()
    thisStar = None
    
data = np.array(data)
print ''
print 'data',len(data)
print 'RVs',RVs.shape
print 'DDRVs',DDRVs.shape
print 'sigmas',sigmas.shape
print 'JDs',JDs.shape


# <codecell>

# #save?
# np.save('data',data)
# np.save('RVs',RVs)
# np.save('DDRVs',DDRVs)
# np.save('sigmas',sigmas)
# np.save('JDs',JDs)


# <headingcell level=3>

# Data post-process 

# <codecell>

#Load?
data=np.load('data.npy')
RVs=np.load('RVs.npy')
DDRVs=np.load('DDRVs.npy')
sigmas=np.load('sigmas.npy')
JDs=np.load('JDs.npy')

# <codecell>

#array to map pivot# to y-position
p2y = pivot_to_y('20aug10042tlm.fits')

#create deltay 2D array of distances between fibres
deltay = np.zeros(np.array(RVs.shape)[[0,0]])
for thisTarget in range(RVs.shape[0]):
    deltay[thisTarget,:] = p2y[data[:,2].astype(int)] - p2y[data[thisTarget,2].astype(int)]

# <codecell>

#create 3D weight array

weights = np.zeros(np.array(RVs.shape)[[0,0,2]]) # Same array for all epochs hence epoch-dimention skipped


for thisTarget in range(RVs.shape[0]): #1 loop per target

    #distances to target (in y-coords)
    mask = np.zeros(np.array(RVs.shape)[[0,0]])
    mask[:,thisTarget] = True
    deltay_mx = np.ma.masked_array(deltay, mask=mask)
    
    #create RV mask to exclude target and stars with RV>3000m/s\
    mask = np.zeros(np.array(RVs.shape)[[0,2]])
    mask[thisTarget] = True
    sigmas_1epc = sigmas[:,0,:]
    sigmas_mx = np.ma.masked_array(sigmas_1epc, mask=mask)
    


    for cam in range(4):
        try:
            a = calibrator_weights(deltay_mx[thisTarget,:].compressed(), sigmas_mx[:,cam].compressed())
            weights[thisTarget,:,cam] = np.insert(a, thisTarget, 0)
        except: 
#             print cam
            pass

# <codecell>

#quad reduced RVs array initialise
quadRVs = np.zeros(RVs.shape) 
diffs = np.zeros(RVs.shape) 

mask = np.zeros(RVs.shape).astype(bool)
RVs_mx = np.ma.masked_array(RVs, mask=mask)

for thisTarget in range(RVs.shape[0]): #1 loop per target
    for epoch in range(RVs.shape[1]):
        for cam in range(RVs.shape[2]):
            
            order = np.argsort(deltay_mx[thisTarget,:])
            quadX = deltay_mx[thisTarget,:][order]
            quadY = RVs_mx[:,epoch,cam][order]
            fRVs,__ = fit_quad([1,1,0], quadX, quadY )
            fittedCurve = quad(quadX, fRVs[0], fRVs[1], fRVs[2])
            
            quadRVs[thisTarget,epoch,cam]=RVs[thisTarget,epoch,cam]-np.sum(weights[thisTarget,:,cam][order]*(RVs[:,epoch,cam][order]-fittedCurve))
            diffs[thisTarget,epoch,cam]=np.sum(weights[thisTarget,:,cam][order]*(RVs[:,epoch,cam][order]-fittedCurve))

quadRVs = np.array(quadRVs)
diffs = np.array(diffs)

# <codecell>

# #array to map pivot# to y-position
# p2y = pivot_to_y('20aug10042tlm.fits')

# #quad reduced RVs array initialise
# quadRVs = np.zeros(RVs.shape)


# for thisTarget in range(RVs.shape[0]): #1 loop per target

#     #distances to target (in y-coords)
#     deltay = p2y[data[:,2].astype(int)] - p2y[data[thisTarget,2].astype(int)]
#     mask = np.zeros(RVs.shape[0]).astype(bool)
#     mask[thisTarget] = True
#     deltay_mx = np.ma.masked_array(deltay, mask=mask)
    
#     #create RV mask to exclude target and stars with RV>3000m/s\
#     mask = np.zeros(RVs.shape).astype(bool)
#     RVs_mx = np.ma.masked_array(RVs, mask=mask)
#     sigmas_mx = np.ma.masked_array(RVs, mask=mask)
#     #     mask[np.unique(np.where(np.abs(RVs)>3000)[0])]=False
# #     mask[[33]]=False
    
#     c = ['b','g','r','cyan']

#     for cam in range(4):
#         weights = calibrator_weights(deltay_mx, sigmas_mx[:,0,cam])
#         weights = np.insert(weights, thisTarget, 0)
#         for epoch in range(5):
            
#             order = np.argsort(deltay_mx)
#             quadX = deltay_mx[order]
#             quadY = RVs_mx[:,epoch,cam][order]
#             fRVs,__ = fit_quad([1,1,0], quadX, quadY )
#             fittedCurve = quad(quadX, fRVs[0], fRVs[1], fRVs[2])
#             quadRVs[thisTarget,epoch,cam]=RVs[thisTarget,epoch,cam]-np.sum(weights[order]*(RVs[:,epoch,cam][order]-fittedCurve))

# #             print RVs[myTarget,epoch,cam]-np.sum(weights*RVs[:,epoch,cam][mask]),RVs[myTarget,epoch,cam],np.sum(weights*RVs[:,epoch,cam][mask])
# #             if cam==0:plt.errorbar(epoch,RVs[myTarget,epoch,cam]-np.sum(weights*RVs[:,epoch,cam][mask]))
# #             plt.scatter(quadX, quadY, marker= '+', c= 'k' , label='Original RVs')
# #             plt.plot(quadX,fittedCurve, label='Quadratic fit')
# #             plt.scatter(quadX,quadY-fittedCurve, c=c[cam], label='Quadratic corrected RVs')
# #             plt.legend(loc=0)
# #             plt.show()
# #             if epoch==3:
# # #                 plt.scatter(deltay[mask][order],
# # #                             RVs[:,epoch,cam][mask][order]-fittedCurve[myTarget],
# # #                             s = 10**3*weights,
# # #                             c= 'r', 
# # #                             label = cam,
# # #                             marker ='o')
# #                 plt.scatter(deltay[mask],
# #                             RVs[:,epoch,cam][mask],
# #                             s = 10**3*weights,
# #                             c= c[cam], 
# #                             label = cam,
# #                             marker ='o')
# #                 plt.scatter(deltay[mask], quadRVs[:,epoch,cam][mask], s = 10**3*weights,c= c[cam], label = cam)
# #                 plt.scatter(deltay[mask], stableRVs[:,epoch,cam][mask], s = 10**3*weights, marker= '+', c= 'k')
# #             order = np.argsort(deltay[mask])
# #             plt.plot(deltay[mask][order], (weights*2000)[order], label=i,c= c[cam])
# #             plt.plot(deltay[mask][order], (sigmas[:,epoch,cam]*100)[order], label=i,c= c[cam])
# #     plt.xticks(data[:,0])
# #     ax.set_xticks(deltay[mask])
# #     ax.set_xticklabels(data[:,0][mask])
# # plt.legend()
# # plt.ylabel('RV[m/s]')
# # plt.xlabel('deltay[px]')
# # plt.show()
# quadRVs = np.array(quadRVs)

# <codecell>

# save?
# np.save('quadRVs',quadRVs)

# <headingcell level=3>

# Plots, sanity checks, results

# <codecell>

c = ['b','g','r','cyan']

# <headingcell level=4>

# Original Data   HD1581 has obj file index 31

# <codecell>

display(Math(r'F(k) = \int_{-\infty}^{\infty} f(x) e^{2\pi i k} dx'))

# <codecell>

#creates images for all targets and all cameras from all original fluxes
for i in starNames:
    filename = i+'.obj'
    filehandler = open(filename, 'r')
    thisStar = pickle.load(filehandler)
    for thisCam, cam in zip(thisStar.exposures.cameras, range(4)):
        for x,y,label in zip(thisCam.wavelengths, thisCam.red_fluxes, thisCam.fileNames):
            plt.plot(x,y, label= label)
        plt.title(thisStar.name)
        plt.legend(loc = 1)
        plt.savefig('plots/'+i+'_'+str(cam))
        plt.close()

# <codecell>

for name,i in zip(fileList, range(len(fileList))):
    print i,name
# filehandler = open(fileList[36], 'r')
# thisStar = pickle.load(filehandler) 

# <headingcell level=4>

# Reduced Data
# HD1581 has starname index 15

# <codecell>

for name,i in zip(starNames, range(len(starNames))):
    print i,name

# <codecell>

filename = 'red_'+starNames[15]+'.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)
thisCam=thisStar.exposures.cameras[2]

# <codecell>

#all red fluxes for the active camera
for x,y,label in zip(thisCam.wavelengths, thisCam.red_fluxes, thisCam.fileNames):
    plt.plot(x,y, label= label)
plt.title(thisStar.name)
plt.legend(loc = 0)
plt.show()

# <codecell>

#all clean fluxes for the active camera
for x,y,label in zip(thisCam.clean_wavelengths, thisCam.clean_fluxes, thisCam.fileNames):
    plt.plot(x,y, label= label)
plt.title(thisStar.name)
plt.legend(loc = 0)
plt.show()

# <codecell>

#all RVs from all cameras for a single target
for thisCam,i in zip(thisStar.exposures.cameras, range(thisStar.exposures.cameras.shape[0])):
    plt.scatter(thisStar.exposures.JDs, thisCam.RVs, c=c[i])
plt.title('HD1581 - RV from median')
plt.ylabel('RV [m/s]')
plt.xlabel('JD')
# plt.legend(loc = 0)
plt.show()

# <codecell>

#all RVs from all cameras for a single target
for thisCam,i in zip(thisStar.exposures.cameras, range(thisStar.exposures.cameras.shape[0])):
    plt.scatter(thisStar.exposures.JDs, thisCam.DDRVs, c=c[i])
plt.title('HD1581 - DDRV from median')
plt.ylabel('RV [m/s]')
plt.xlabel('JD')
# plt.legend(loc = 0)
plt.show()

# <headingcell level=4>

# final data

# <codecell>

for name,i in zip(data[:,0], range(len(data))):
    print i,name

# <codecell>

# all RVs for all camera for a single target
thisTarget = 31 #HD1581
for i in range(4):
    plt.scatter(JDs, RVs[31,:,i], c = c[i])
plt.show()

# <codecell>

# all DDRVs for all camera for a single target
thisTarget = 31 #HD1581
for i in range(4):
    plt.scatter(JDs, DDRVs[31,:,i], c = c[i])
plt.show()

# <codecell>

DDRVs

# <headingcell level=1>

# Random stuff....

# <codecell>

myTarget=36

# <codecell>

deltay = p2y[data[:,2].astype(int)] - p2y[data[myTarget,2].astype(int)]
mask = np.zeros(RVs.shape[0]).astype(bool)
mask[myTarget] = True
deltay_mx = np.ma.masked_array(deltay, mask=mask)

#create RV mask to exclude target and stars with RV>3000m/s\
mask = np.zeros(RVs.shape).astype(bool)
RVs_mx = np.ma.masked_array(RVs, mask=mask)
sigmas_mx = np.ma.masked_array(sigmas, mask=mask)
sigmas_mx.mask[myTarget,:,:] = True
# print calibrator_weights(deltay_mx.compressed(), sigmas_mx[:,0,cam].compressed()).shape
a = calibrator_weights(deltay_mx, sigmas_mx[:,0,cam])
a = np.insert(a, myTarget, 0)
plt.plot( a)
plt.show()

# <codecell>

quadRVs

# <codecell>

plt.scatter (JDs, quadRVs[36,:,1])
plt.show()

# <codecell>

mask

# <codecell>

fRVs,__ = optimize.curve_fit(quad, quadX, quadY, p0 = [-0.001,-0.001,quadY[np.where(deltay==np.min(np.abs(deltay)))[0][0]]], )
plt.scatter( quadX, quadY)
smoothX = np.linspace(np.min(quadX), np.max(quadX))
fittedCurve = quad(quadX, fRVs[0], fRVs[1], fRVs[2])
plt.plot(quadX,fittedCurve)
plt.scatter(quadX,quadY*fittedCurve, c='r')
plt.show()
print 'params',fRVs

# <codecell>

plt.scatter(JDs,quadRVs[30,:,0], c='r', label = 'stable star (observed)')
plt.scatter(JDs,quadRVs[36,:,1], c='g', label = 'HD285507 (observed)')
start_day = 2456889.500000 # The Julian date for CE  2014 August 20 00:00:00.0 UT  (10am australia)
end_day = 2456895.500000 #The Julian date for CE  2014 August 26 00:00:00.0 UT (10am australia)

days = np.linspace(start_day, end_day)  - 2400000

P = 6.0881
peri_arg = 182
peri_time = 2456257.5- 2400000
K1 =125.8
RV = K1* np.sin((days-peri_time)/P*2*np.pi + peri_arg/360*2*np.pi )
plt.plot(days, RV, linewidth = 1, label = 'HD285507' )
plt.legend(loc=0)
plt.xlabel('JD')
plt.ylabel('RV [m/s]')
plt.show()

# <codecell>

# for i in range(quadRVs.shape[0]):
print np.min((np.nansum(np.abs(quadRVs[:,:,:-1]), axis=1)/4))
print (np.nansum(np.abs(quadRVs), axis=1)/4)


# <codecell>

plt.plot(sigmas.flatten())
plt.show()

# <headingcell level=2>

# Plots

# <codecell>

def plot_all_spec_cams(thisStar):    #All spectra for all cameras
    
    fig, ax = plt.subplots(2,2, sharey='all')
    
    # ax.set_yticks(thisStar.exposures.JDs)
    # ax.set_ylim(np.min(thisStar.exposures.JDs)-1,np.min(thisStar.exposures.JDs)+1)
    for cam in range(4):
        thisCam = thisStar.exposures.cameras[cam]
        fileNames =  thisCam.fileNames
        nFluxes = thisCam.wavelengths.shape[0]
        ax[0,0].set_yticks(np.arange(0,nFluxes))
        ax[0,0].set_ylim(-1,nFluxes)
    
        for i in np.arange(nFluxes):
            d, f = thisCam.clean_wavelengths[i], thisCam.clean_fluxes[i]
            if cam ==0:
                ax[0,0].plot(d, f+i, 'b')
            elif cam==1:
                ax[0,1].plot(d, f+i, 'g')
            elif cam==2:
                ax[1,0].plot(d, f+i, 'r')
            elif cam==3:
                ax[1,1].plot(d, f+i, 'cyan')
        #         ax.plot(d, f+thisStar.exposures.JDs[i], 'k')
    
    plt.xlabel('Wavelength [Ang]')
    plt.title(thisStar.name+' - Camera '+str(cam+1))
    ax[0,0].set_yticklabels(fileNames)
    plt.show()

# <codecell>

RVs1 = np.load('RVs1.npy') 
RVs2 = np.load('RVs2.npy') 
RVs3 = np.load('RVs3.npy') 
RVs4 = np.load('RVs4.npy') 
JDs = np.load('JDs.npy') 

# <codecell>

#Plots RVs, baryvels for all 4 cameras
plt.title('Average of all decorrelated targets')
mask = np.abs(quadRVs)<3000
quadRVs[-mask]=np.nan
# for i in range(RVs.shape[0]):
    # plt.plot(thisStar.exposures.JDs, thisStar.exposures.rel_baryVels, label = 'Barycentric Vel. ')
try:
#         plt.scatter(JDs, stableRVs[i,:,0], label = 'Blue', color ='b' )
#     plt.scatter(JDs, np.median(quadRVs[:,:,0],axis=0), label = 'Blue', color ='b' )
    plt.errorbar(JDs, np.nanmean(quadRVs[:,:,0],axis=0), yerr=np.nanstd(quadRVs[:,:,0],axis=0), label = 'Blue', color ='b' )
#     plt.errorbar(JDs, np.nanmean(RVs[:,:,0],axis=0), yerr=np.nanstd(RVs[:,:,0],axis=0), label = 'Blue', color ='b' )
    pass
except:pass
try:
#     plt.scatter(JDs, stableRVs[i,:,1], label = 'Green', color ='g' )
#     plt.scatter(JDs, np.median(quadRVs[:,:,1],axis=0), label = 'Green', color ='g' )
    plt.errorbar(JDs, np.nanmean(quadRVs[:,:,1],axis=0), yerr=np.nanstd(quadRVs[:,:,1],axis=0), label = 'Green', color ='g' )
#     plt.errorbar(JDs, np.nanmean(RVs[:,:,1],axis=0), yerr=np.nanstd(RVs[:,:,1],axis=0), label = 'Green', color ='g' )
    pass
except:pass
try:
#     plt.scatter(JDs, stableRVs[i,:,2], label = 'Red', color ='r' )
#     plt.scatter(JDs, np.median(quadRVs[:,:,2],axis=0), label = 'Red', color ='r' )
    plt.errorbar(JDs, np.nanmean(quadRVs[:,:,2],axis=0), yerr=np.nanstd(quadRVs[:,:,2],axis=0), label = 'Red', color ='r' )
#     plt.errorbar(JDs, np.nanmean(RVs[:,:,2],axis=0), yerr=np.nanstd(RVs[:,:,2],axis=0), label = 'Red', color ='r' )
    pass
except:pass
try:
#     plt.scatter(JDs, stableRVs[i,:,3], label = 'IR', color ='cyan' )
#     plt.scatter(JDs, np.median(stableRVs[:,:,3],axis=0), label = 'IR', color ='cyan' )
    plt.errorbar(JDs, np.nanmean(quadRVs[:,:,3],axis=0), yerr=np.nanstd(quadRVs[:,:,3],axis=0), label = 'IR', color ='cyan' )
    pass
except:pass

# start_day = 56889.000000 # The Julian date for CE  2014 August 20 00:00:00.0 UT  (10am australia)
# end_day = 56895.000000 #The Julian date for CE  2014 August 26 00:00:00.0 UT (10am australia)

# days = np.linspace(start_day, end_day) 

# K1=26100
# peri_time = 19298.85
# P=4.8202
# peri_arg=269.3
# RV = K1* np.sin( (days-peri_time)/P*2*np.pi - peri_arg/360*2*np.pi )
# plt.plot(days, RV, linewidth = 1, label = 'rhoTuc' )

plt.xlabel('JD')
plt.ylabel('RV [m/s]')
plt.legend(loc=0)
plt.show()

# <codecell>

mask = np.abs(quadRVs)<3000
plt.plot(quadRVs[mask].flatten())
plt.show()

# <codecell>

# np.save('RVs1',RVs1) 
# np.save('RVs2',RVs2) 
# np.save('RVs3',RVs3) 
# np.save('RVs4',RVs4) 
# np.save('JDs',JDs) 

# <codecell>

#Plots RVs, baryvels. Single star, 4 cameras
plt.title(filehandler.name[:-4])
# plt.plot(thisStar.exposures.JDs, thisStar.exposures.rel_baryVels, label = 'Barycentric Vel. ')

thisCam = thisStar.exposures.cameras[0]
# RVMask = thisStar.exposures.my_data_mask
# RVMask = ((thisStar.exposures.my_data_mask) & (np.abs(thisCam.RVs)<50000))
RVMask = thisCam.safe_flag
plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'Blue', color ='b' )
# plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'Blue', color ='b' )

thisCam = thisStar.exposures.cameras[1]
# RVMask = thisStar.exposures.my_data_mask
# RVMask = np.abs(thisCam.RVs)<1e6
RVMask = thisCam.safe_flag
plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'Green' , color ='g')
# plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.RVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'Green' , color ='g')

thisCam = thisStar.exposures.cameras[2]
# RVMask = thisStar.exposures.my_data_mask
# RVMask = np.abs(thisCam.RVs)<2e5
RVMask = thisCam.safe_flag
plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'Red' , color ='r')
# plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.RVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'Red' , color ='r')

thisCam = thisStar.exposures.cameras[3]
# RVMask = thisStar.exposures.my_data_mask
# RVMask = ((thisStar.exposures.my_data_mask) & (np.abs(thisCam.RVs)<20000))
RVMask = thisCam.safe_flag
plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'IR', color ='cyan' )
# plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.RVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'IR', color ='cyan' )

# start_day = 56889.000000 # The Julian date for CE  2014 August 20 00:00:00.0 UT  (10am australia)
# end_day = 56895.000000 #The Julian date for CE  2014 August 26 00:00:00.0 UT (10am australia)

# days = np.linspace(start_day, end_day) 

# K1=26100
# peri_time = 19298.85
# P=4.8202
# peri_arg=269.3
# RV = K1* np.sin( (days-peri_time)/P*2*np.pi - peri_arg/360*2*np.pi )
# plt.plot(days, RV, linewidth = 1, label = 'rhoTuc' )

plt.xlabel('JD')
plt.ylabel('RV [m/s]')
plt.legend(loc=3)
plt.show()

# <codecell>

thisCam = thisStar.exposures.cameras[0]
# for i in range(5):
print thisCam.Ps
print (thisCam.Ps[0,0]-thisCam.clean_wavelengths[0].shape[0]/2)
print (-20480)*3000

# <codecell>

#plots all data
thisCam = thisStar.exposures.cameras[2]
for i in [3]:
    
    fig = plt.gcf()
    fig.suptitle(filehandler.name[:-4]+' - t0 vs t'+str(i)+' - RV='+str(thisCam.RVs[i])+' m/s', fontsize=14)

    plt.subplot(221)
    plt.title('Clean Flux')
    plt.plot(thisCam.clean_wavelengths[i][::50],thisCam.clean_fluxes[i][::50])
    plt.xlabel('Wavelength [ang]')
    plt.ylabel('Flux [Counts]')
    plt.legend(loc=0)
    
    plt.subplot(223)
    plt.title('Clean Flux - Detail')
    plt.plot(thisCam.clean_wavelengths[0],thisCam.clean_fluxes[0], label = 't0 Flux')
    plt.plot(thisCam.clean_wavelengths[i],thisCam.clean_fluxes[i], label = 'Epoch '+str(i))
#     plt.axis((4859,4864, -1,1))
    plt.xlabel('Wavelength [ang]')
    plt.ylabel('Flux [Counts]')
    plt.legend(loc=0)

    plt.subplot(222)
    plt.title('Cross Correlation Result')
    plt.plot((thisCam.clean_wavelengths[i]-np.median(thisCam.clean_wavelengths[i]))/thisCam.clean_wavelengths[i]*const.c,thisCam.CCCurves[i]/np.max(thisCam.CCCurves[i]))
    plt.plot((thisCam.clean_wavelengths[i]-np.median(thisCam.clean_wavelengths[i]))/thisCam.clean_wavelengths[i]*const.c,
             gaussian(range(len(thisCam.clean_wavelengths[i])), thisCam.Ps[i][0], thisCam.Ps[i][1]), label = 'gaussian fit') 
    plt.xlabel('RV [m/s]')
    plt.legend(loc=0)
    
    plt.subplot(224)
    plt.title('Cross Correlation Result - Detail')
    plt.plot((thisCam.clean_wavelengths[i]-np.median(thisCam.clean_wavelengths[i]))/thisCam.clean_wavelengths[i]*const.c,thisCam.CCCurves[i]/np.max(thisCam.CCCurves[i]))
    plt.plot((thisCam.clean_wavelengths[i]-np.median(thisCam.clean_wavelengths[i]))/thisCam.clean_wavelengths[i]*const.c,
             gaussian(range(len(thisCam.clean_wavelengths[i])), thisCam.Ps[i][0], thisCam.Ps[i][1]), label = 'gaussian fit') 
    plt.axis((-8000, 8000,0.5,1.1) )
    plt.xlabel('RV [m/s]')
    plt.legend(loc=2)    

    plt.tight_layout()
    plt.show()

# <codecell>


# <codecell>

#plots all CC fitted gausian results\n",
# for i in range(CCCurves.shape[0]):\n",
thisCam = thisStar.exposures.cameras[0]
for i in range(10):
    plt.plot((thisCam.clean_wavelengths[i]-np.median(thisCam.clean_wavelengths[i]))/thisCam.clean_wavelengths[i]*const.c,thisCam.CCCurves[i]/np.max(thisCam.CCCurves[i]))
#     plt.plot((thisCam.clean_wavelengths[i]-np.median(thisCam.clean_wavelengths[i]))/thisCam.clean_wavelengths[i]*const.c,
#              gaussian(range(len(thisCam.clean_wavelengths[i])), thisCam.Ps[i][0], thisCam.Ps[i][1])) 
    plt.xlabel('RV [m/s]')
plt.show()

# <codecell>

#Debuging CC - resampling and cleaning tests
flux1 = thisStar.exposures.cameras[0].red_fluxes[0]
lambda1 = thisStar.exposures.cameras[0].wavelengths[0]
lambda1Clean_10, flux1Clean_10 = clean_flux(flux1, 1, lambda1)
# lambda1Clean_10, flux1Clean_10 = clean_flux(flux1, xDef, lambda1)
# lambda1Clean_100, flux1Clean_100 = clean_flux(flux1, 100, lambda1)\n",
plt.plot(lambda1,flux1/np.max(flux1), label= 'Reduced')
# plt.plot(lambda1Clean_1,flux1Clean_1)
plt.plot(lambda1Clean_10,flux1Clean_10, label= 'Clean')
# plt.plot(lambda1Clean_100,flux1Clean_100)
plt.title('Reduced and Clean flux')
plt.xlabel('Wavelength [Ang.]')
plt.ylabel('Relatuve Flux')
plt.legend(loc=0)
plt.show()

# <codecell>

#Saves
# delattr(thisStar.exposures.cameras[0],'clean_fluxes')
# delattr(thisStar.exposures.cameras[1],'clean_fluxes')
# delattr(thisStar.exposures.cameras[2],'clean_fluxes')
# delattr(thisStar.exposures.cameras[3],'clean_fluxes')
# delattr(thisStar.exposures.cameras[0],'clean_wavelengths')
# delattr(thisStar.exposures.cameras[1],'clean_wavelengths')
# delattr(thisStar.exposures.cameras[2],'clean_wavelengths')
# delattr(thisStar.exposures.cameras[3],'clean_wavelengths')
# file_pi = open(filehandler.name, 'w') 
# pickle.dump(thisStar, file_pi) 
# file_pi.close()
# filehandler.close()
# thisStar = None

# <codecell>

1/10e-6/3600

# <codecell>


