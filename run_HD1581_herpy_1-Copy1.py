
# coding: utf-8

# In[31]:

# import pyfits as pf
# import pylab as plt
# from scipy import optimize
# from scipy.signal import medfilt, find_peaks_cwt
# from scipy.ndimage.filters import minimum_filter, maximum_filter, median_filter, convolve
# from scipy.ndimage.measurements import label
# import numpy as np

import myHerpyTools as MHT
reload(MHT)


# In[2]:

cd /Users/Carlos/Documents/HERMES/reductions/new_start_6.5/HD1581/


# In[3]:

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


# In[4]:

#Opening files
flat = MHT.openFile(flatFileName)
arc =  MHT.openFile(arcFileName)
obj =  MHT.openFile(objFileName)


# In[5]:

flat_flat = MHT.make_flat_flat(flat)


# In[6]:

#Check results
# plt.imshow(flat_flat)
# plt.show()


# In[7]:

# #Thesis plots. Skip for processing
# i=20
# singleCol = flat_flat[:,i].copy()
# singleMinEnv = convolve(minimum_filter(singleCol,15),[.2,.2,.2,.2,.2])
# singleMin = singleCol - singleMinEnv

# singleMax = convolve(maximum_filter(singleMin,15),[.2,.2,.2,.2,.2]) 

# fixer = convolve(singleMax, np.ones(200)/200)
# singleMax[singleMax<fixer*.5] = fixer[singleMax<fixer*.5]*.5
# singleColFlat = singleMin.copy()/singleMax.copy()
# singleMax += singleMinEnv

# # plt.plot(singleCol)
# # plt.plot(singleMinEnv) 
# # plt.plot(singleMax) 

# singleColFlat_bin = singleColFlat.copy()
# singleColFlat_bin[singleColFlat>.3] = 1
# singleColFlat_bin[singleColFlat<.3] = 0

# flat_flat_bin = flat_flat.copy()
# flat_flat_bin[:,i] = singleColFlat_bin

# plt.title("Normalised Fibres and Binary Mask")
# plt.xlabel("Pixel Index")
# plt.ylabel("Intensity")
# plt.plot(singleColFlat)
# # plt.plot(flat_flat_bin[:,i]) 
# plt.fill_between(range(flat_flat_bin.shape[0]),flat_flat_bin[:,i], alpha =0.3, color ='black')
# plt.show()


# In[8]:

flat_flat_bin = MHT.make_flat_flat_bin(flat_flat)


# In[ ]:

# #Check results
# plt.plot(flat_flat_bin[:,102])
# plt.show()


# In[ ]:




# In[ ]:

# plt.imshow(out_array)
# plt.show()


# In[ ]:

# np.max(out_array)


# In[12]:

fibre_centroids = MHT.make_fibre_centroids(flat_flat_bin)


# In[16]:

# fibre_centroids


# In[15]:

# MHT.plt.imshow(MHT.np.isnan(fibre_centroids))
# MHT.plt.show()


# In[17]:

# for i in range(397):
#     print np.sum(np.isnan(fibre_centroids[3,:]))


# In[ ]:

#line to remove 251 and 252 that have nans
# fibre_centroids = np.delete(fibre_centroids,251,0) #2 times for epoch0
# fibre_centroids = np.delete(fibre_centroids,372,0) 3 times for epoch1
# np.sum(np.isnan(fibre_centroids),1)


# In[20]:

fibrePolys = MHT.make_fibrePolys(fibre_centroids)


# In[23]:

tramlines = MHT.make_tramlines(fibre_centroids, fibrePolys)


# In[25]:

#find vertical shift
shift = MHT.find_vertical_shift(flat, arc) #result to be subtracted to the tramlines (1st array in the CC...)
tramlines_shifted = tramlines - shift


# In[28]:

# shift


# In[27]:

# #gaussian fit results
# plt.plot(x,y)
# # plt.plot(x_dense,gaussian(x_dense,p[0],p[1]))
# plt.show()


# In[ ]:

def extract(tramlines_shifted, data):
    extracted = np.ones(tramlines_shifted.shape)*np.nan
    for fibre in range(tramlines_shifted.shape[0]):
        extracted[fibre] = sum_extract(fibre,tramlines_shifted, data, 4)
    return extracted


# In[33]:

extracted_arc = MHT.extract(tramlines_shifted, arc)


# In[36]:

extracted_obj = MHT.extract(tramlines_shifted, obj)


# In[35]:

# MHT.plt.plot(MHT.np.median(extracted_arc,axis=0))
# MHT.plt.show()


# In[37]:

MHT.plt.plot(extracted_arc[170])
MHT.plt.show()


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



