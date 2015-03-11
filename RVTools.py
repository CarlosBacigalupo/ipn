# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pickle
import pylab as plt
from scipy import interpolate, signal, optimize, constants
import pyfits as pf
import sys

# <codecell>

def clean_flux(wavelength, flux, thisCam, xDef = 1, medianRange = 0):
    '''Clean a 1D spectrum. 
    
    Parameters
    ----
    xDef : int or None, optional
        Coeficient to resample. Final array will be flux.shape[0]*xDef long. 
        
    medianRange : int, optional
        Number of pixels to median over. 0 will skip this step. Optional.

    '''
        
    #median outliers
    if medianRange>0:
        fluxMed = signal.medfilt(flux,medianRange)
        w = np.where(abs((flux-fluxMed)/np.maximum(fluxMed,50)) > 0.4)
        for ix in w[0]:
            flux[ix] = fluxMed[ix]

    if ((wavelength[-np.isnan(flux)].shape[0]>0) &  (flux[-np.isnan(flux)].shape[0]>0)):
        
        #flatten curve by fitting a 3rd order poly
        fFlux = optimize.curve_fit(cubic, wavelength[-np.isnan(flux)], flux[-np.isnan(flux)], p0 = [1,1,1,1])
        fittedCurve = cubic(wavelength, fFlux[0][0], fFlux[0][1], fFlux[0][2], fFlux[0][3])
        flux = flux/fittedCurve-1
        
        #apply tukey
        flux = flux * tukey(0.1, len(flux))

        #resample
        if (xDef>1):
            fFlux = interpolate.interp1d(wavelength, flux) 
            wavelength = np.linspace(min(wavelength), max(wavelength),len(wavelength)*xDef)
            flux = fFlux(wavelength)

    else: #if not enough data return NaNs
        if (xDef>1):
            wavelength = np.linspace(min(wavelength), max(wavelength),len(wavelength)*xDef)
            flux = np.ones(wavelength.shape[0])*np.nan
        
    return wavelength, flux


# <codecell>

def cubic(x,a,b,c,d):
    '''
    Cubic function
    '''
    return a*x**3+b*x**2+c*x+d

# <codecell>

def tukey(alpha, N):
    '''Creates a tukey function
    
    
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

# <codecell>

#Find the common range of wl for all cameras
def find_max_wl_range(thisStar):
    for thisCam in thisStar.exposures.cameras:
        if np.nansum(thisCam.wavelengths - thisCam.wavelengths[0,:])==0:
            print 'WL aligned'
        else:
            print 'WL NOT aligned'

        mask = np.isnan(thisCam.red_fluxes)
        collapsed_mask = np.sum(mask, axis=0)==0
        single_wl =  thisCam.wavelengths[0][collapsed_mask]
        thisCam.wavelengths = np.reshape(np.tile(single_wl, thisCam.wavelengths.shape[0]), 
                         ((thisCam.wavelengths.shape[0], single_wl.shape[0])))
        full_mask =  np.reshape(np.tile(collapsed_mask, thisCam.wavelengths.shape[0]), thisCam.red_fluxes.shape)
        thisCam.red_fluxes =  np.reshape(thisCam.red_fluxes[full_mask], (thisCam.red_fluxes.shape[0], np.sum(collapsed_mask)))

# <codecell>

#Create cross correlation curves wrt epoch 0
def RVs_CC_t0(thisStar, xDef = 1, CCReferenceSet = 0, printDetails=False, corrHWidth=10):

    print '#############################################################################################################'
    print '#############################################################################################################'
    print '#############################################################################################################'

#     validDates = np.all([np.nansum(thisCam.red_fluxes,1).astype(bool) for thisCam in thisStar.exposures.cameras],0)
    
    for j,thisCam in enumerate(thisStar.exposures.cameras):
        RVs = []
        sigmas = [] 
        Qs = []
        SNRs = []

        validDates = np.nansum(thisCam.red_fluxes,1).astype(bool)
        
        print 'Camera',j,validDates
        
        if len(np.arange(len(validDates))[validDates])>0:
            CCReferenceSet = np.arange(len(validDates))[validDates][0]
        else:
            CCReferenceSet = 0
            
            
        lambda1, flux1 = clean_flux(thisCam.wavelengths[CCReferenceSet], thisCam.red_fluxes[CCReferenceSet], thisCam)
        print 'Reference wl,fl sum', np.nansum(lambda1), np.nansum(flux1)
        print 
        
        plts = 0    
        for i, JD in enumerate(thisStar.exposures.JDs):
            print 'Set,MJD',i,JD
            lambda2, flux2 = clean_flux(thisCam.wavelengths[i], thisCam.red_fluxes[i], thisCam)
            print 'This wl,fl sum', np.nansum(lambda2), np.nansum(flux2)

            if validDates[i]==True:
                CCCurve = signal.fftconvolve(flux1[-np.isnan(flux1)], flux2[-np.isnan(flux2)][::-1], mode='same')
#                 CCCurve = signal.fftconvolve(flux1, flux2[::-1], mode='same')
#                 if i <5:
#                     plt.plot(flux1)
#                     plt.plot(flux2)
#                     plt.plot(CCCurve)
#                     plt.show()
                print 'max_idx, len(CCCurve) =',np.where(CCCurve==max(CCCurve)), CCCurve.shape
                try:
                    corrMax = np.where(CCCurve==max(CCCurve))[0][0]
                except: 
                    print 'cc',CCCurve,
#                     if plts<5:
#                         plts+=1
#                         plt.plot(flux1)
#                         plt.plot(flux2)
#                         plt.plot(CCCurve)
#                         plt.show()

                p_guess = [corrMax,corrHWidth]
                x_mask = np.arange(corrMax-corrHWidth, corrMax+corrHWidth+1)
                if max(x_mask)<len(CCCurve):
    #                 try:
    #                 print '4 params',p_guess, x_mask, np.sum(x_mask), CCCurve.shape
                    p = fit_gaussian(p_guess, CCCurve[x_mask], np.arange(len(CCCurve))[x_mask])[0]
                    print 'p result',p
                    if np.modf(CCCurve.shape[0]/2.0)[0]>1e-5:
                        pixelShift = (p[0]-(CCCurve.shape[0]-1)/2.) #odd number of elements
                    else:
                        pixelShift = (p[0]-(CCCurve.shape[0])/2.) #even number of elements
    #                 except:
    #                     pixelShift = 0

                    thisQ, thisdRV = QdRV(thisCam.wavelengths[i], thisCam.red_fluxes[i])
                    print  ' thisQ, thisdRV',thisQ, thisdRV

                    mid_px = thisCam.wavelengths.shape[1]/2
                    dWl = (thisCam.wavelengths[i,mid_px+1]-thisCam.wavelengths[i,mid_px]) / thisCam.wavelengths[i,mid_px]
                    RV = dWl * pixelShift * constants.c 
                    print 'RV',RV

                    SNR = np.median(thisCam.red_fluxes[i])/np.std(thisCam.red_fluxes[i])
                else:
                    R = 0
                    thisQ = 0
                    thisdRV = 0
                    RV = 0
                    print 'Invalid data point'

            else:
#                 if i==CCReferenceSet:
#                     print 'The CC reference set is not present. Can\'t continue. Launch again with different reference set.'
#                     sys.exit()
                SNR = 0
                thisQ = 0
                thisdRV = 0
                RV = 0
                print 'Invalid data point'

            print ''

            SNRs.append(SNR)
            Qs.append(thisQ)
            sigmas.append(thisdRV)
            RVs.append(RV)


                
        thisCam.sigmas = np.array(sigmas)
        thisCam.Qs = np.array(Qs)
        thisCam.RVs = np.array(RVs)
        thisCam.SNRs = np.array(SNRs)

# <codecell>

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

def extract_HERMES_wavelength(fileName):

	a = pf.open(fileName)

	CRVAL1 = a[0].header['CRVAL1'] # / Co-ordinate value of axis 1                    
	CDELT1 = a[0].header['CDELT1'] #  / Co-ordinate increment along axis 1             
	CRPIX1 = a[0].header['CRPIX1'] #  / Reference pixel along axis 1                   
	
	#Creates an array of offset wavelength from the referece px/wavelength
	Lambda = CRVAL1 - (CRPIX1 - (np.arange(int(CRPIX1)*2)) -1)* CDELT1

	return Lambda

# <codecell>

def pivot_to_y(ref_file):
     
    a = pf.getdata(ref_file)
    
    return a[:,200]

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

