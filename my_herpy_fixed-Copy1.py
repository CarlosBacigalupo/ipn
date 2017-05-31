
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

def openFile(fileName):
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



# In[ ]:

def fithOrder(thisPoly, thisRange):
    result = thisPoly[0]*thisRange**5
    result += thisPoly[1]*thisRange**4
    result += thisPoly[2]*thisRange**3
    result += thisPoly[3]*thisRange**2
    result += thisPoly[4]*thisRange**1
    result += thisPoly[5]*thisRange**0
    
    return result


# In[ ]:

def find_vertical_shift(flat, arc):
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
################
##Need to SUBTRACT the result of the gaussian fit to make the 1st curve be like the second (i.e the traces be like the arc)


# In[ ]:

def sum_extract(fibre, tramlines, image, numPx):
    
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


# In[ ]:

# flatFileName = '0_20aug/1/20aug10034.fits'
# arcFileName = '0_20aug/1/20aug10052.fits'
# objFileName = '0_20aug/1/20aug10053.fits'

# flatFileName = '1_21aug/1/21aug10047.fits'
# arcFileName = '1_21aug/1/21aug10046.fits'
# objFileName = '1_21aug/1/21aug10041.fits'

# flatFileName = '2_22aug/1/22aug10032.fits'
# arcFileName = '2_22aug/1/22aug10031.fits'
# objFileName = '2_22aug/1/22aug10036.fits'

flatFileName = '3_24aug/1/24aug10053.fits'
arcFileName = '3_24aug/1/24aug10054.fits'
objFileName = '3_24aug/1/24aug10058.fits'

# flatFileName = '4_25aug/1/25aug10039.fits'
# arcFileName = '4_25aug/1/25aug10043.fits'
# objFileName = '4_25aug/1/25aug10044.fits'


# In[ ]:

#Opening files
flat = openFile(flatFileName)
arc =  openFile(arcFileName)
obj =  openFile(objFileName)


# In[ ]:

#Flat fielding
flat_mf = medfilt(flat, [3,9])
flat_1d = np.sum(flat_mf,axis =0)
flat_per = np.percentile(flat_1d, 90)
flat_1d_norm = flat_1d/flat_per
flat_flat = flat_mf / flat_1d_norm[None,:]


# In[ ]:

#Check results
# plt.imshow(flat_flat)
# plt.show()


# In[ ]:

#Thesis plots. Skip for processing
i=20
singleCol = flat_flat[:,i].copy()
singleMinEnv = convolve(minimum_filter(singleCol,15),[.2,.2,.2,.2,.2])
singleMin = singleCol - singleMinEnv

singleMax = convolve(maximum_filter(singleMin,15),[.2,.2,.2,.2,.2]) 

fixer = convolve(singleMax, np.ones(200)/200)
singleMax[singleMax<fixer*.5] = fixer[singleMax<fixer*.5]*.5
singleColFlat = singleMin.copy()/singleMax.copy()
singleMax += singleMinEnv

# plt.plot(singleCol)
# plt.plot(singleMinEnv) 
# plt.plot(singleMax) 

singleColFlat_bin = singleColFlat.copy()
singleColFlat_bin[singleColFlat>.3] = 1
singleColFlat_bin[singleColFlat<.3] = 0

flat_flat_bin = flat_flat.copy()
flat_flat_bin[:,i] = singleColFlat_bin

plt.title("Normalised Fibres and Binary Mask")
plt.xlabel("Pixel Index")
plt.ylabel("Intensity")
plt.plot(singleColFlat)
# plt.plot(flat_flat_bin[:,i]) 
plt.fill_between(range(flat_flat_bin.shape[0]),flat_flat_bin[:,i], alpha =0.3, color ='black')
plt.show()


# In[ ]:

#Convert flat_flat to binary for tracing
flat_flat_bin = flat_flat.copy()

for i in range(flat_flat.shape[1]):
    print i,
    singleCol = flat_flat[:,i].copy()
    
    singleMinEnv = convolve(minimum_filter(singleCol,15),[.2,.2,.2,.2,.2])
    singleMin = singleCol - singleMinEnv
    
    singleMax = convolve(maximum_filter(singleMin,15),[.2,.2,.2,.2,.2])

    fixer = convolve(singleMax, np.ones(200)/200)
    singleMax[singleMax<fixer*.5] = fixer[singleMax<fixer*.5]*.5
    singleColFlat = singleMin/singleMax
    
    singleColFlat[singleColFlat>.3] = 1
    singleColFlat[singleColFlat<.3] = 0
    
    flat_flat_bin[:,i] = singleColFlat
    


# In[ ]:

#Check results
plt.plot(flat_flat_bin[:,102])
plt.show()


# In[ ]:

out_array, n = label(flat_flat_bin, np.ones((3,3)))
print n,'fibres'
# n-=2 # fibres 252 and 253 are not good for HD1581 epoch 0 


# In[ ]:

plt.imshow(out_array)
plt.show()


# In[ ]:

np.max(out_array)


# In[ ]:

#create centroid array
fibres = n
cols = out_array.shape[1]
fibre_centroids = np.ones((rows,cols))*np.nan

for fibre in range(1,fibres+1):
    wRows, wCols = np.where(out_array==fibre)
    print fibre,
    for col in range(max(wCols)+1):
        fibre_centroids[fibre-1, col] = np.average(wRows[wCols==col])


# In[ ]:

plt.imshow(np.isnan(fibre_centroids))
plt.show()


# In[ ]:

for i in range(397):
    print np.sum(np.isnan(fibre_centroids[3,:]))


# In[ ]:

#line to remove 251 and 252 that have nans
# fibre_centroids = np.delete(fibre_centroids,251,0) #2 times for epoch0
# fibre_centroids = np.delete(fibre_centroids,372,0) 3 times for epoch1
np.sum(np.isnan(fibre_centroids),1)


# In[ ]:

#create polynomials from centroids
fibrePolys = np.ones((fibre_centroids.shape[0],6))*np.nan
for y,fibre in enumerate(fibre_centroids):
    fibrePolys[y-1,:] = np.polyfit(range(fibre.shape[0]),fibre,5)
    if np.sum(np.isnan(fibrePolys[y-1,:]))>0:
        print 'Found nan in fibre',y
        print 'Fibre values',fibre[np.isnan(fibre)]
        print


# In[ ]:

#create tramlines from polynomials
tramlines = (np.ones(fibre_centroids.shape)*np.nan)[:-1]
thisRange = np.arange(fibre_centroids.shape[1])
for i,thisPoly in enumerate(fibrePolys[1:]):
    tramlines[i] = fithOrder(thisPoly,thisRange)


# In[ ]:

#find vertical shift
shift = find_vertical_shift(flat, arc) #result to be subtracted to the tramlines (1st array in the CC...)
tramlines_shifted = tramlines - shift


# In[ ]:

shift


# In[ ]:

#gaussian fit results
plt.plot(x,y)
# plt.plot(x_dense,gaussian(x_dense,p[0],p[1]))
plt.show()


# In[ ]:

extracted_arc = np.ones(tramlines_shifted.shape)*np.nan
for fibre in range(tramlines_shifted.shape[0]):
    extracted_arc[fibre] = sum_extract(fibre,tramlines_shifted, arc, 4)


# In[ ]:

extracted_obj = np.ones(tramlines_shifted.shape)*np.nan
for fibre in range(tramlines_shifted.shape[0]):
    extracted_obj[fibre] = sum_extract(fibre,tramlines_shifted, obj, 4)


# In[ ]:

plt.plot(np.median(extracted_obj,axis=1))
plt.show()


# In[ ]:

plt.plot(extracted_arc[170])
plt.show()


# In[ ]:

#At this point we have the tramlines created, arc and obj extracted. No rebinning or wl solution yet...


# In[ ]:

#initial pixel adjustment
thisFibre = 200

print "Fibre:",thisFibre

halfCCRange = 10
halfSmallCCRange = 4

masterArc = extracted_arc[thisFibre].copy()

lineTemplate = np.zeros(extracted_arc.shape[1])
lineList = np.loadtxt('linelist_blue.txt')
lineTemplate[lineList[:,0].astype(int)]=1

CCCurve = np.correlate(masterArc, lineTemplate, mode='full')

y = CCCurve[int(CCCurve.shape[0]/2.)+1-halfCCRange:int(CCCurve.shape[0]/2.)+1+halfCCRange+1]
x = np.arange(-halfCCRange,halfCCRange+1)

maxIdx = np.where(y==np.max(y))[0][0]
print x[maxIdx]


# plt.plot(masterArc)
# adjLineList = lineList.copy()
# adjLineList[:,0] += x[maxIdx]
# np.savetxt('linelist_blue_v2.txt',adjLineList)
# plt.scatter(adjLineList[:,0], np.ones(adjLineList.shape[0]))
# plt.show()


# In[ ]:

#FULL LOOP

wlSolutions = []
wlErrors = []
wlPolys = []
for thisFibre in range(extracted_arc.shape[0])[:]:
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
#         plt.plot(x_dense,gaussian(x_dense,p[0],p[1]))
#         plt.plot(firstSliceX,firstSliceY/np.max(firstSliceY))
#         plt.title(maxIdx)
# #         plt.plot(masterArc[thisLineWl[0]-5:thisLineWl[0]+5])
#         plt.show()


# In[ ]:

import scipy.constants as const

wlErrorsRVs = wlErrors/np.tile(adjLineList[:,1],[wlErrors.shape[0],1])*const.c

stdRVs = np.std(wlErrorsRVs,axis = 1)
# stdRVs2 = np.std(wlErrorsRVs_General_distrib,axis = 1)

plt.plot(stdRVs, '.')
# plt.plot(stdRVs2, '.')
plt.grid()
plt.show()
#237 !!!!


# In[ ]:

plt.plot(wlSolutions[170],extracted_arc[170])
# plt.plot(wlSolutions[20],extracted_arc[20])
plt.show()


# In[ ]:

thisArray = extracted_arc[237].copy()
thisArray[thisArray<20]=0
thisPeaks = find_peaks_cwt(thisArray, np.arange(1,2))
print len(thisPeaks)


# In[ ]:

plt.scatter(thisPeaks,extracted_arc[237][thisPeaks])
plt.plot(extracted_arc[237])
plt.show()


# In[ ]:

#extend line list to all bumps

masterArc = extracted_arc[237]
bigLineList = []
for i, thisPeak in enumerate(thisPeaks):
#         print 'Searching for wl',thisLineWl[1],'in px',thisLineWl[0],

    firstSliceX = np.arange(thisPeak-5,thisPeak+6).astype(int)
    firstSliceY = masterArc[firstSliceX]
    maxIdx =  firstSliceX[np.where(firstSliceY==np.max(firstSliceY))[0][0]]

    secondSliceX = np.arange(maxIdx-5,maxIdx+6).astype(int)
    secondSliceY = masterArc[secondSliceX]      

#         p,_ = fit_gaussian([maxIdx,1.], secondSliceY, secondSliceX )
    p,_ = fit_flexi_gaussian([maxIdx,1., 2., np.max(secondSliceY), 0], secondSliceY, secondSliceX )

#         print 'Found',p
    goodPxValue = p[0]

#     adjLineList[i,0] = goodPxValue

#         x_dense = np.linspace(np.min(secondSliceX),np.max(secondSliceX))
#         plt.plot(secondSliceX,secondSliceY)
#         plt.plot(x_dense,flexi_gaussian(x_dense,p[0],p[1],p[2],p[3],p[4]))
#         plt.title(goodPxValue)
#         plt.show()

#     print adjLineList

# a = np.polyfit(adjLineList[:,0], adjLineList[:,1], 5)
    x = fithOrder(wlPolys[237], goodPxValue)
# err = fithOrder(a, adjLineList[:,0]) - adjLineList[:,1]

    bigLineList.append((goodPxValue,x))
# wlErrors.append(err)
# wlSolutions.append(x)

bigLineList = np.array(bigLineList)
# wlPolys = np.array(wlPolys)
# wlErrors = np.array(wlErrors)
# wlSolutions = np.array(wlSolutions)
#         plt.plot(x_dense,gaussian(x_dense,p[0],p[1]))
#         plt.plot(firstSliceX,firstSliceY/np.max(firstSliceY))
#         plt.title(maxIdx)
# #         plt.plot(masterArc[thisLineWl[0]-5:thisLineWl[0]+5])
#         plt.show()


# In[ ]:

plt.plot(range(wlSolutions[237].shape[0]), wlSolutions[237])
plt.scatter(bigLineList[:,0], bigLineList[:,1])
plt.show()


# In[ ]:

plt.plot(wlSolutions[40], extracted_arc[40])
plt.scatter(bigLineList[:,1],np.ones(110)*20)
plt.show()


# In[ ]:

np.savetxt('bigLineList.txt',bigLineList)


# In[ ]:

# _,ind = np.unique(bigLineList[:,1], return_index=True)
# bigLineList =  bigLineList[ind]
# print bigLineList


# In[ ]:

#find differences to other fibres

# bigLineList = np.loadtxt('bigLineList.txt')
bigLineList = reducedbigLineList
# lineLocations = np.ones((extracted_arc.shape[0],bigLineList.shape[0]))*np.nan
lineLocations = np.ones(bigLineList.shape[0])*np.nan

for thisFibre in range(extracted_arc.shape[0])[170:171]:
    print 'Fibre',thisFibre,

    halfCCRange = 15

    thisArc = extracted_arc[thisFibre].copy()
    thisObj = extracted_obj[thisFibre].copy()
    thisWlSolution = wlSolutions[thisFibre].copy()
    

    for i, thisWl in enumerate(bigLineList):
#         print 'Searching for wl',thisWl
        diffArray = np.abs(thisWlSolution-thisWl)
#         print thisWl, thisWlSolution, diffArray 
        wlPx = np.where(diffArray==np.min(diffArray))[0][0]
#         print wlPx, thisWlSolution[wlPx-1:wlPx+1]

        thisSlice = np.arange(wlPx-6,wlPx+7).astype(int)

        firstSliceX = thisWlSolution[thisSlice]
        firstSliceY = thisArc[thisSlice]
        maxIdx =  thisSlice[np.where(firstSliceY==np.max(firstSliceY))[0][0]]
#         print maxIdx
        
#         plt.plot(firstSliceX,firstSliceY)
#         plt.show()

        
        thisSecondSlice = np.arange(maxIdx-5,maxIdx+6).astype(int)
        secondSliceX = thisWlSolution[thisSecondSlice]
        secondSliceY = thisArc[thisSecondSlice]      
    
# #         p,_ = fit_gaussian([maxIdx,1.], secondSliceY, secondSliceX )
        p,_ = fit_flexi_gaussian([thisWlSolution[maxIdx],.2, 2.8, np.max(secondSliceY), 0], secondSliceY, secondSliceX )
        
#         print 'Found',p[0]
        goodWlValue = p[0]
        
#         lineLocations[thisFibre,i] = goodWlValue
        lineLocations[i] = goodWlValue

#         adjLineList[i,0] = goodPxValue
        
#         x_dense = np.linspace(np.min(secondSliceX),np.max(secondSliceX))
#         plt.plot(secondSliceX,secondSliceY)
#         plt.plot(x_dense,flexi_gaussian(x_dense,p[0],p[1],p[2],p[3],p[4]))
# # #         plt.title(goodPxValue)
#         plt.show()

# #     print adjLineList
    
#     a = np.polyfit(adjLineList[:,0], adjLineList[:,1], 5)
#     x = fithOrder(a, np.arange(4095))
#     err = fithOrder(a, adjLineList[:,0]) - adjLineList[:,1]

#     wlPolys.append(a)
#     wlErrors.append(err)
#     wlSolutions.append(x)
    
# wlPolys = np.array(wlPolys)
# wlErrors = np.array(wlErrors)
# wlSolutions = np.array(wlSolutions)
# #         plt.plot(x_dense,gaussian(x_dense,p[0],p[1]))
# #         plt.plot(firstSliceX,firstSliceY/np.max(firstSliceY))
# #         plt.title(maxIdx)
# # #         plt.plot(masterArc[thisLineWl[0]-5:thisLineWl[0]+5])
# #         plt.show()


# In[ ]:

# means = np.mean(lineLocations,axis=0)
# offsets = lineLocations - np.tile(bigLineList[:,1],[388,1])
offsets = lineLocations - bigLineList

lineLocations.shape, offsets.shape


# In[ ]:

plt.plot(bigLineList,offsets.transpose(),'.')
# plt.plot(np.arange(bigLineList[:,1].shape[0])[filter_all],offsets.transpose()[filter_all],'.')
# plt.plot(bigLineList[:,1],offsets.transpose(),'.')
# plt.plot(offsets.transpose(),'.')
# plt.plot(bigLineList[:,1],std, color ='r')
plt.grid()
plt.show()


# In[ ]:

np


# In[ ]:

std = np.nanstd(offsets, axis =0)


# In[ ]:

plt.plot(std,'.')
plt.show()


# In[ ]:

filter1 = np.abs(np.nanmedian(offsets,axis=0))<0.06
np.sum(filter1)


# In[ ]:

np.sum(np.abs(offsets)<0.6,axis=0)


# In[ ]:

offsets[np.abs(offsets)>0.6]=np.nan


# In[ ]:

filter2 = std<.12


# In[ ]:

filter_all = filter1 & filter2 &filter3


# In[ ]:

filter1


# In[ ]:

bad = np.array([19 ,29, 46, 51, 55, 59, 69, 74, 75, 79, 85])


# In[ ]:

filter3 = np.ones(filter1.shape).astype(bool)
filter3[bad] = False


# In[ ]:

np.savetxt('reducedbigLineList.txt',bigLineList[:,1][filter_all])


# In[ ]:

reducedbigLineList = bigLineList[:,1][filter_all]


# In[ ]:

reducedbigLineList


# In[ ]:

plt.plot(wlSolutions[170], extracted_obj[170])
plt.show()


# In[ ]:

# np.savetxt('HD1581_1.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
# np.savetxt('ThXe_1.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())


# In[ ]:

# np.savetxt('HD1581_0.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
np.savetxt('ThXe_0.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())


# In[ ]:

np.savetxt('HD1581_2.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
np.savetxt('ThXe_2.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())


# In[ ]:

np.savetxt('HD1581_3.txt',np.vstack((wlSolutions[170], extracted_obj[170])).transpose())
np.savetxt('ThXe_3.txt',np.vstack((wlSolutions[170], extracted_arc[170])).transpose())


# In[ ]:



