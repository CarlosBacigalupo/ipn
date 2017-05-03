
# coding: utf-8

# In[3]:

import pyfits as pf
import pylab as plt
from scipy import optimize
from scipy.signal import medfilt, find_peaks_cwt
from scipy.ndimage.filters import minimum_filter, maximum_filter, median_filter, convolve
from scipy.ndimage.measurements import label
import numpy as np


# In[1]:

cd /Users/Carlos/Documents/HERMES/reductions/myherpy/


# In[5]:

def openFile(fileName):
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
    
    gain0_2000  = thisFile[0].header['RO_GAIN']
    gain2000_4000  = thisFile[0].header['RO_GAIN1']

    thisData = thisFile[0].data

    bias0_2000 = np.median(thisData[3:2052,4099:-3])
    bias2000_4000 = np.median(thisData[2059:-3,4099:-3])

    thisData = thisData[:,:4095]

    thisData[:2055] -= bias0_2000
    thisData[2055:] -= bias2000_4000
    
    return thisData


def fithOrder(thisPoly, thisRange):
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


# In[135]:

# flatFileName = '0_20aug/1/20aug10034.fits'
# arcFileName = '0_20aug/1/20aug10052.fits'
# objFileName = '0_20aug/1/20aug10053.fits'

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


# In[136]:

flat = openFile(flatFileName)
arc =  openFile(arcFileName)
obj =  openFile(objFileName)


# In[137]:

flat_mf = medfilt(flat, [3,9])
flat_1d = np.sum(flat_mf,axis =0)
flat_per = np.percentile(flat_1d, 90)
flat_1d_norm = flat_1d/flat_per
flat_flat = flat_mf / flat_1d_norm[None,:]


# In[138]:

for i in range(flat_flat.shape[1]):
    singleCol = flat_flat[:,i]
    singleMin = singleCol - convolve(minimum_filter(singleCol,15),[.2,.2,.2,.2,.2])
    singleMax = convolve(maximum_filter(singleMin,15),[.2,.2,.2,.2,.2])

    fixer = convolve(singleMax, np.ones(200)/200)
    singleMax[singleMax<fixer*.5] = fixer[singleMax<fixer*.5]*.5
    singleColFlat = singleMin/singleMax

    singleColFlat[singleColFlat>.3] = 1
    singleColFlat[singleColFlat<.3] = 0
    
    flat_flat[:,i] = singleColFlat


# In[139]:

out_array, n = label(flat_flat, np.ones((3,3)))


# In[140]:

#create centroid array
fibre_centroids = np.ones((n,out_array.shape[1]))*np.nan
for col in range(out_array.shape[1]):
    testCol = out_array[:,col]
    for fibre in range(n):
        fibre_centroids[fibre,col] = np.average(np.where(testCol==fibre+1)[0])
#         if np.sum(np.isnan(fibre_centroids[fibre,col]))>0:
#             print 'Found nan',np.where(testCol==fibre+1)[0], fibre+1



# In[141]:

fibrePolys = np.ones((fibre_centroids.shape[0],6))*np.nan
for y,fibre in enumerate(fibre_centroids):
    fibrePolys[y,:] = np.polyfit(range(fibre.shape[0]),fibre,5)
#     if np.sum(np.isnan(fibrePolys[y,:]))>0:
#         print 'Found nan in fibre',y
#         print 'Fibre values',fibre
#         print


# In[142]:

#create tramlines
tramlines = (np.ones(fibre_centroids.shape)*np.nan)[:-1]
thisRange = np.arange(fibre_centroids.shape[1])
for i,thisPoly in enumerate(fibrePolys[1:]):
    tramlines[i] = fithOrder(thisPoly,thisRange)


# In[143]:

shift = find_vertical_shift(flat, arc) #result to be subtracted to the tramlines (1st array in the CC...)
tramlines_shifted = tramlines - shift


# In[144]:

extracted_arc = np.ones(tramlines_shifted.shape)*np.nan
for fibre in range(tramlines_shifted.shape[0]):
    extracted_arc[fibre] = sum_extract(fibre,tramlines_shifted, arc, 4)


# In[145]:

extracted_obj = np.ones(tramlines_shifted.shape)*np.nan
for fibre in range(tramlines_shifted.shape[0]):
    extracted_obj[fibre] = sum_extract(fibre,tramlines_shifted, obj, 4)


# In[146]:

wlSolutions = []
wlErrors = []
wlPolys = []
for thisFibre in range(extracted_arc.shape[0])[:]:
#     print 'Fibre',thisFibre
    print thisFibre,

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

#     print 'Shift', thisShift

    adjLineList = lineList.copy()
    adjLineList[:,0] += thisShift

    for i, thisLineWl in enumerate(adjLineList):
#         print 'Searching for wl',thisLineWl[1],'in px',thisLineWl[0],
        
        firstSliceX = np.arange(thisLineWl[0]-5,thisLineWl[0]+6).astype(int)
        firstSliceY = masterArc[firstSliceX]
        maxIdx =  firstSliceX[np.where(firstSliceY==np.max(firstSliceY))[0][0]]
        
        secondSliceX = np.arange(maxIdx-5,maxIdx+6).astype(int)
        secondSliceY = masterArc[secondSliceX]      
        
#         p,_ = fit_gaussian([maxIdx,1.], secondSliceY, secondSliceX )
        p,_ = fit_flexi_gaussian([maxIdx,1., 2., np.max(secondSliceY), 0], secondSliceY, secondSliceX )
        
#         print 'Found',p
        goodPxValue = p[0]

        adjLineList[i,0] = goodPxValue
        
#         x_dense = np.linspace(np.min(secondSliceX),np.max(secondSliceX))
#         plt.plot(secondSliceX,secondSliceY)
#         plt.plot(x_dense,flexi_gaussian(x_dense,p[0],p[1],p[2],p[3],p[4]))
#         plt.title(goodPxValue)
#         plt.show()

#     print adjLineList
    
    a = np.polyfit(adjLineList[:,0], adjLineList[:,1], 5)
    x = fithOrder(a, np.arange(4095))
    err = fithOrder(a, adjLineList[:,0]) - adjLineList[:,1]

    wlPolys.append(a)
    wlErrors.append(err)
    wlSolutions.append(x)
    
wlPolys = np.array(wlPolys)
wlErrors = np.array(wlErrors)
wlSolutions = np.array(wlSolutions)


# In[147]:

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



