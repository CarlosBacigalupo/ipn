import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

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


wavelengths = np.loadtxt("wavelengths.txt")
fluxes = np.loadtxt("fluxes.txt")

fig, ax = plt.subplots()
for i in xrange(29):
    try:
        d, f = fit_continuum(wavelengths[i], fluxes[i])
    except: continue
    else:
        ax.plot(d,f+i, 'k')

plt.show()
