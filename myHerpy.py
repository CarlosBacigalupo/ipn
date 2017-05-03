
# coding: utf-8

# In[ ]:

import pyfits as pf
import pylab as plt
from scipy import optimize
from scipy.signal import medfilt, find_peaks_cwt
from scipy.ndimage.filters import minimum_filter, maximum_filter, median_filter, convolve
from scipy.ndimage.measurements import label
import numpy as np


# In[ ]:

cd /Users/Carlos/Documents/HERMES/reductions/myherpy/HD1581/


# In[ ]:

a = np.arange(10)
a[6:]


# In[ ]:

def openFile(fileName): #TODO use gain to scale adus
    '''
    Open a HERMES fits file. Subtracts bias from overscan

    Parameters
    ----------
        fileName: input file name

    Returns
    -------
        thisData : 2D array with the bias subtracted image of the first HDU.
    
    '''
    thisFile = pf.open(fileName)

    print thisFile[0].header['OBJECT']
    
    #Readout amplifier (inverse) gain (e-/ADU) 
    gain0_2000  = thisFile[0].header['RO_GAIN']
    gain2000_4000  = thisFile[0].header['RO_GAIN1']

    # Readout noise (electrons)
    noise0_2000  = thisFile[0].header['RO_NOISE']
    noise2000_4000  = thisFile[0].header['RO_NOIS1']

    thisData = thisFile[0].data
    
    print 'raw array shape',thisData.shape
    
    bias0_2000 = np.median(thisData[3:2052,4099:-3])
    bias2000_4000 = np.median(thisData[2059:-3,4099:-3])

    thisData = thisData[:,:4095]

    thisData[:2056] -= bias0_2000
    thisData[2056:] -= bias2000_4000
    
    thisData[:2056] *= gain0_2000
    thisData[2056:] *= gain2000_4000

    print 'output array shape',thisData.shape
    print

    return thisData


def fifthOrder(thisPoly, thisRange):
    '''
    Evaluates a 5th order polybomial at the specified range

    Parameters
    ----------
        thisPoly: Array. 6 x polynomila coefficients

        thisRange: Array. Range of values for the independent variable

    Returns
    -------
        result : polynomial evaluated over the range
    
    '''

    result = thisPoly[0]*thisRange**5
    result += thisPoly[1]*thisRange**4
    result += thisPoly[2]*thisRange**3
    result += thisPoly[3]*thisRange**2
    result += thisPoly[4]*thisRange**1
    result += thisPoly[5]*thisRange**0
    
    return result

def find_vertical_shift(flat, arc):
    '''
    Calculates the vertical shift between the flat and the arc. 
    Needs to SUBTRACT the result of the gaussian fit to the traces 
    to make the 1st curve be like the second (i.e the traces be like the arc)


    Parameters
    ----------
    flat: 2-D Array. Flat image

    arc: 2-D Array. Arc Imag


    Returns
    -------
    shift : vertical pixel shift calculated
    
    '''
    
    CCTotal = 0
    for column in range(flat.shape[1]):
        thisFlatCol = flat[:,column]
        thisArcCol = arc[:,column]
        CCCurve = np.correlate(thisFlatCol, thisArcCol, mode='full')
        CCTotal += CCCurve

    y = CCTotal[int(CCTotal.shape[0]/2.)+1-5:int(CCTotal.shape[0]/2.)+1+4]
    y /=np.max(y)
    x = np.arange(-4,5)
    x_dense = np.linspace(-4,4)
    p,_ = fit_gaussian([1,3.],y,x )
    shift = p[0]
    
    return shift


def sum_extract(fibre, tramlines, image, numPx):
    '''
    Extracts the flux from a given fibre from an image using the tramline map. 

    Parameters
    ----------
    fibre: Fibre number to be extracted

    tramlines: 2-D Array. Fibre centroids

    image: 2-D Array. Image of the flux to be extracted

    numPx: Number of pixels to extract on each side of the centroid


    Returns
    -------
    flux : Extracted flux
    
    '''
    
    flux = np.ones(tramlines.shape[1])*np.nan
#     flux1 = np.ones(tramlines.shape[1])*np.nan
#     flux2 = np.ones(tramlines.shape[1])*np.nan
    
    for i,thisCentroid in enumerate(tramlines[fibre]):
#         print thisCentroid
        try:
            fullPx = image[ int(thisCentroid)-numPx : int(thisCentroid)+numPx+1 , i]
            flux[i] = np.sum(fullPx) - fullPx[0]*(thisCentroid%1) - fullPx[-1]*(1-thisCentroid%1)
#         flux1[i] = fullPx[0]*(thisCentroid%1)
#         flux2[i] = fullPx[-1]*(1-thisCentroid%1)
        except:
            print fibre, 'falied'
            print thisCentroid, 'centroid found in index',i
            break
#             print fibre
    return flux


def nosum_extract_img(fibre, tramlines, image, numPx):
    '''
    Extracts the flux from a given fibre from an image using the tramline map. NO SUM 

    Parameters
    ----------
    fibre: Fibre number to be extracted

    tramlines: 2-D Array. Fibre centroids

    image: 2-D Array. Image of the flux to be extracted

    numPx: Number of pixels to extract on each side of the centroid


    Returns
    -------
    flux : Extracted flux
    
    '''
    
    flux = np.ones((numPx*2+1,tramlines.shape[1]))*np.nan
#     flux1 = np.ones(tramlines.shape[1])*np.nan
#     flux2 = np.ones(tramlines.shape[1])*np.nan
    
    for i,thisCentroid in enumerate(tramlines[fibre]):
#         print thisCentroid
#         for j, range(numPx*2+1);
        try:
            fullPx = image[ int(thisCentroid)-numPx : int(thisCentroid)+numPx+1 , i]
            flux[:,i] = fullPx
            flux[0,i] -= fullPx[0]*(thisCentroid%1)
    #         print thisCentroid%1
            flux[-1,i] -= fullPx[-1]*(1-thisCentroid%1)
#             flux[i] = np.sum(fullPx) - fullPx[0]*(thisCentroid%1) - fullPx[-1]*(1-thisCentroid%1)
#         flux1[i] = fullPx[0]*(thisCentroid%1)
#         flux2[i] = fullPx[-1]*(1-thisCentroid%1)
        except:
            print fibre, 'falied'
            print thisCentroid, 'centroid found in index',i
#             break
# #             print fibre
    return flux

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


# In[ ]:

thisFile = pf.open('0_20aug/1/20aug10034.fits')
thisFile[0].header


# In[ ]:

flatFileName = '0_20aug/1/20aug10034.fits'
arcFileName = '0_20aug/1/20aug10052.fits'
objFileName = '0_20aug/1/20aug10053.fits'

# flatFileName = '1_21aug/1/21aug10047.fits'
# arcFileName = '1_21aug/1/21aug10046.fits'
# objFileName = '1_21aug/1/21aug10041.fits'
# objFileName = '1_21aug/1/21aug10042.fits'
# objFileName = '1_21aug/1/21aug10043.fits'

# flatFileName = '2_22aug/1/22aug10032.fits'
# arcFileName = '2_22aug/1/22aug10031.fits'
# objFileName = '2_22aug/1/22aug10036.fits'
# objFileName = '2_22aug/1/22aug10037.fits'
# objFileName = '2_22aug/1/22aug10038.fits'

# flatFileName = '3_24aug/1/24aug10053.fits'
# arcFileName = '3_24aug/1/24aug10054.fits'
# objFileName = '3_24aug/1/24aug10058.fits'
# objFileName = '3_24aug/1/24aug10059.fits'
# objFileName = '3_24aug/1/24aug10060.fits'
# objFileName = '3_24aug/1/24aug10061.fits'
# objFileName = '3_24aug/1/24aug10062.fits'

# flatFileName = '4_25aug/1/25aug10039.fits'
# arcFileName = '4_25aug/1/25aug10043.fits'
# objFileName = '4_25aug/1/25aug10044.fits'
# objFileName = '4_25aug/1/25aug10045.fits'
# objFileName = '4_25aug/1/25aug10046.fits'


# # Open files

# In[ ]:

flat = openFile(flatFileName)
arc =  openFile(arcFileName)
obj =  openFile(objFileName)


# In[ ]:

#stuff to check the open files
# import pylab as plt
# plt.imshow(obj)
# plt.plot(np.sum(obj, axis=1))
# # plt.plot(np.sum(obj[2000:2100, :200], axis=1))
# # plt.plot(np.sum(obj[2000:2100, 3000:3200], axis=1))
# plt.show()


# # Flat operations

# ### Initial flat handling

# In[ ]:

flat_mf = medfilt(flat, [3,9])
flat_1d = np.sum(flat_mf,axis =0) 
flat_per = np.percentile(flat_1d, 90)
flat_1d_norm = flat_1d/flat_per

flat_flat = flat_mf / flat_1d_norm[None,:]


# In[ ]:

#THis is an example of what the singleMin line does
# singleCol = flat_flat[:,10]
# a = minimum_filter(singleCol,15)
# b = convolve(a, [.2,.2,.2,.2,.2])
# plt.plot(singleCol)
# plt.plot(a)
# plt.plot(b)
# plt.show()


# In[ ]:

#Example of singleMax
# singleMin = singleCol - convolve(minimum_filter(singleCol,15),[.2,.2,.2,.2,.2])
# singleMax = convolve(maximum_filter(singleMin,15),[.2,.2,.2,.2,.2])
# plt.plot(singleMin)    
# plt.plot(singleMax)
# plt.plot(fixer)
# singleColFlat = singleMin/singleMax
# plt.plot(singleColFlat)
# plt.show()


# In[ ]:

#turn to binary
for i in range(flat_flat.shape[1]):
    
    singleCol = flat_flat[:,i] #isolates a single column
    
    #removes a somoothen (convolved) array of the minima in a 15px range from singleCol
    singleMin = singleCol - convolve(minimum_filter(singleCol,15),[.2,.2,.2,.2,.2])
    
    #smoothen (convolved) maxima in a 15px range
    singleMax = convolve(maximum_filter(singleMin,15),[.2,.2,.2,.2,.2])
    
    #removes the gaps between maxima (where lower than 1/2 of the smoothered (fixer) version)
    fixer = convolve(singleMax, np.ones(200)/200)
    singleMax2 = singleMax.copy()
    singleMax[singleMax<fixer*.5] = fixer[singleMax<fixer*.5]*.5
    
    #ratio btween max and min
    singleColFlat = singleMin/singleMax

    #Set cut-off value (0.3)
    singleColFlat[singleColFlat>.3] = 1
    singleColFlat[singleColFlat<.3] = 0
    
    #write back to flat
    flat_flat[:,i] = singleColFlat

    
#flat_flat becames a single-bit mape of the tram regions


# ### Find goups of 1s and 0s in flat_flat

# In[ ]:

#looks for groups (label features) in flat_flat using a 3by3 stamp. 
#out_array is the list of features, one int per group, same shape than flat_flat
# n is the number of groups (labels) 
out_array, n = label(flat_flat, np.ones((3,3)))


# In[ ]:

out_array


# ### Create the centroids array. (nFibres, 4095)

# In[ ]:

#create centroid array
fibre_centroids = np.ones((n,out_array.shape[1]))*np.nan
for col in range(out_array.shape[1]):
    testCol = out_array[:,col]
    for fibre in range(n):
        fibre_centroids[fibre,col] = np.average(np.where(testCol==fibre+1)[0])
#         print np.average(np.where(testCol==fibre+1)[0])
#         print fibre_centroids
#         if np.sum(np.isnan(fibre_centroids[fibre,col]))>0:
#             print 'Found nan',np.where(testCol==fibre+1)[0], fibre+1



# In[ ]:

fibrePolys.shape


# ### Create 5th deg polynomials to map out the centroids. (nFibres, 6)

# In[ ]:

fibrePolys = np.ones((fibre_centroids.shape[0],6))*np.nan
for y,fibre in enumerate(fibre_centroids):
    fibrePolys[y,:] = np.polyfit(range(fibre.shape[0]),fibre,5)
#     if np.sum(np.isnan(fibrePolys[y,:]))>0:
#         print 'Found nan in fibre',y
#         print 'Fibre values',fibre
#         print


# ### Create tramlines from 5th deg polynomials (nFibres, 4095)
# 

# In[ ]:

#create tramlines
tramlines = np.ones(fibre_centroids.shape)*np.nan
thisRange = np.arange(fibre_centroids.shape[1])
for i,thisPoly in enumerate(fibrePolys):
    tramlines[i] = fifthOrder(thisPoly,thisRange)


# In[ ]:

plt.imshow(out_array, cmap="gray")
# a = out_array
# a[out_array==1]=100
# plt.imshow(a)
# for i in range(tramlines.shape[0]):
plt.plot(tramlines[170,:])
plt.show()


# # Arc Operations

# ### Find Vertical Shift between flat and arc (shift flat to be like arc)

# In[ ]:

shift = find_vertical_shift(flat, arc) #result to be subtracted to the tramlines (1st array in the CC...)
tramlines_shifted = tramlines - shift
 
shift = find_vertical_shift(flat, obj) #result to be subtracted to the tramlines (1st array in the CC...)
tramlines_shifted_obj = tramlines - shift
 
#why this is important for the arc but not for the obj?


# In[ ]:

plt.imshow(flat_sub_n, cmap="gray")
# plt.plot(np.sum(arc_sub, axis=0))
# for i in range(tramlines.shape[0]):
i=171
# plt.plot(tramlines[i,:], c='r')
# plt.plot(tramlines_shifted[i,:], c='b')
# plt.plot(tramlines_shifted_obj[i,:], c='g')
plt.show()


# In[ ]:

flat_sub = flat[np.min(np.floor(tramlines[171,:]))-3:np.max(np.floor(tramlines[171,:]))+5, :]
arc_sub = arc[np.min(np.floor(tramlines[171,:]))-3:np.max(np.floor(tramlines[171,:]))+5, :]
obj_sub = obj[np.min(np.floor(tramlines[171,:]))-3:np.max(np.floor(tramlines[171,:]))+5, :]


# In[ ]:

# plt.plot(flat_sub.transpose())
plt.plot(flat_sub[0,:])
xp = np.linspace(0, flat_sub.shape[1], 4000)
plt.plot(a(xp))

plt.show()


# In[ ]:

from scipy import interpolate


# In[ ]:

a = interpolate.UnivariateSpline(range(flat_sub.shape[1]),flat_sub[0,:], s=1000 )


# In[ ]:

a(range(flat_sub.shape[1]))


# In[ ]:

flat_sub_n = flat_sub/np.sum(flat_sub, axis=0)
flat_sub_n,flat_sub


# ### Sum Extract arc fluxes using shifted tram lines

# In[ ]:

(9, tramlines_shifted.shape[1],tramlines_shifted.shape[0])


# In[ ]:

#TODO justify sum vs optimal extraction (Horne 1986, Robertson 92? )
# extracted_arc = np.ones(tramlines_shifted.shape)*np.nan
nPix = 4
extracted_flat_img = np.ones((nPix*2+1, tramlines_shifted.shape[1],tramlines_shifted.shape[0]))*np.nan
extracted_arc_img = np.ones((nPix*2+1, tramlines_shifted.shape[1],tramlines_shifted.shape[0]))*np.nan
extracted_obj_img = np.ones((nPix*2+1, tramlines_shifted.shape[1],tramlines_shifted.shape[0]))*np.nan

for fibre in range(tramlines_shifted.shape[0]):
#     extracted_arc[fibre] = sum_extract(fibre,tramlines_shifted, arc, 4)
    extracted_flat_img[:,:,fibre] = nosum_extract_img(fibre,tramlines_shifted, flat, nPix)
    extracted_arc_img[:,:,fibre] = nosum_extract_img(fibre,tramlines_shifted, arc, nPix)
    extracted_obj_img[:,:,fibre] = nosum_extract_img(fibre,tramlines_shifted, obj, nPix)


# ##Normalise the flat column by column

# In[ ]:

extracted_flat_img_flat = extracted_flat_img / np.sum(extracted_flat_img, axis=0)
extracted_arc_img_norm = extracted_arc_img / extracted_flat_img_flat
extracted_obj_img_norm = extracted_obj_img / extracted_flat_img_flat


# In[ ]:

plt.plot(extracted_flat_img_flat[:,:,171].transpose())
plt.show()


# In[ ]:

plt.imshow(a[:,:100])
plt.clim(-10,30)
plt.show()


# In[ ]:

a = nosum_extract_img(171,tramlines_shifted, arc, 4)


# In[ ]:

plt.plot


# ### Sum extract object fluxes using shifted tramlines

# In[ ]:

extracted_obj = np.ones(tramlines_shifted.shape)*np.nan
for fibre in range(tramlines_shifted.shape[0]):
    extracted_obj[fibre] = sum_extract(fibre,tramlines_shifted, obj, 4)


# In[ ]:

pwd


# In[ ]:

wlSolutions = []
wlErrors = []
wlPolys = []

for thisFibre in range(extracted_arc.shape[0])[:5]:
    print 
    print 'Fibre',thisFibre

    halfCCRange = 15

    masterArc = extracted_arc[thisFibre].copy()

    lineTemplate = np.zeros(extracted_arc.shape[1])
    lineList = np.loadtxt('linelist_blue_v2.txt')
    lineTemplate[lineList[:,0].astype(int)]=1

    CCCurve = np.correlate(masterArc, lineTemplate, mode='full')

    y = CCCurve[int(CCCurve.shape[0]/2.)+1-halfCCRange:int(CCCurve.shape[0]/2.)+1+halfCCRange+1]
    x = np.arange(-halfCCRange,halfCCRange+1)

    maxIdx = np.where(y==np.max(y))[0][0]
    thisShift = x[maxIdx]

    print 'Shift', thisShift

    #these 2 lines adjust the px value assigned in lineList based on the shift found above
    adjLineList = lineList.copy()
    adjLineList[:,0] += thisShift

    for i, thisLineWl in enumerate(adjLineList):
        print 'Searching for wl',thisLineWl[1],'in px',thisLineWl[0],
        
        #first run looks for max in master arc +-5 px from nominal corrected px position in adjLineList
        firstSliceX = np.arange(thisLineWl[0]-5,thisLineWl[0]+6).astype(int)
        firstSliceY = masterArc[firstSliceX]
        maxIdx =  firstSliceX[np.where(firstSliceY==np.max(firstSliceY))[0][0]]
        
        #second run slices +-5 px from max
        secondSliceX = np.arange(maxIdx-5,maxIdx+6).astype(int)
        secondSliceY = masterArc[secondSliceX]      
        
        #gaussian fit on the slice found
#         p,_ = fit_gaussian([maxIdx,1.], secondSliceY, secondSliceX )
        p,_ = fit_flexi_gaussian([maxIdx,1., 2., np.max(secondSliceY), 0], secondSliceY, secondSliceX )
        print 'Found',p
        goodPxValue = p[0]
        
        #replace pixel value for gaussian fit result
        adjLineList[i,0] = goodPxValue
        
#         x_dense = np.linspace(np.min(secondSliceX),np.max(secondSliceX))
#         plt.plot(secondSliceX,secondSliceY)
#         plt.plot(x_dense,flexi_gaussian(x_dense,p[0],p[1],p[2],p[3],p[4]))
#         plt.title(goodPxValue)
#         plt.show()

    print adjLineList
    
    a = np.polyfit(adjLineList[:,0], adjLineList[:,1], 5)
    x = fifthOrder(a, np.arange(4095))
    err = fifthOrder(a, adjLineList[:,0]) - adjLineList[:,1]

    wlPolys.append(a)
    wlErrors.append(err)
    wlSolutions.append(x)
    
wlPolys = np.array(wlPolys)
wlErrors = np.array(wlErrors)
wlSolutions = np.array(wlSolutions)


# In[ ]:

# np.savetxt('HD1581_1_42.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
# np.savetxt('ThXe_1_42.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())

# np.savetxt('HD1581_1_43.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
# np.savetxt('ThXe_1_43.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())

# np.savetxt('HD1581_2_37.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
# np.savetxt('ThXe_2_37.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())

# np.savetxt('HD1581_2_38.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
# np.savetxt('ThXe_2_38.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())

# np.savetxt('HD1581_3_59.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
# np.savetxt('ThXe_3_59.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())

# np.savetxt('HD1581_3_60.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
# np.savetxt('ThXe_3_60.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())

# np.savetxt('HD1581_3_61.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
# np.savetxt('ThXe_3_61.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())

# np.savetxt('HD1581_3_62.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
# np.savetxt('ThXe_3_62.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())

# np.savetxt('HD1581_4.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
# np.savetxt('ThXe_4.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())

# np.savetxt('HD1581_4_45.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
# np.savetxt('ThXe_4_45.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())

np.savetxt('HD1581_4_46.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
np.savetxt('ThXe_4_46.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())


# In[ ]:



