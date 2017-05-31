
# coding: utf-8

# In[2]:

import pyfits as pf
import pylab as plt
from scipy import optimize
from scipy.signal import medfilt, find_peaks_cwt
from scipy.ndimage.filters import minimum_filter, maximum_filter, median_filter, convolve
from scipy.ndimage.measurements import label
import numpy as np


# In[ ]:

# cd /Users/Carlos/Documents/HERMES/reductions/myherpy/HD1581/


# In[ ]:

#Opens a file and subtracts bias from overscann
#In: filename
#out: thisData (data array)
def openFile(fileName):
    thisFile = pf.open(fileName)

    print thisFile[0].header['OBJECT']
    
    gain0_2000  = thisFile[0].header['RO_GAIN']
    gain2000_4000  = thisFile[0].header['RO_GAIN1']

    thisData = thisFile[0].data

    bias0_2000 = np.median(thisData[3:2052,4099:-3])
    bias2000_4000 = np.median(thisData[2059:-3,4099:-3])

    thisData = thisData[:,:4096]

    thisData[:2055] -= bias0_2000
    thisData[2055:] -= bias2000_4000
    
    return thisData



# In[ ]:

#Cleans and flattens the flat
#in: raw flat
#out: flattened flat
def make_flat_flat(flat):
    #Flat fielding
    flat_mf = medfilt(flat, [3,9])
    flat_1d = np.sum(flat_mf,axis =0)
    flat_per = np.percentile(flat_1d, 90)
    flat_1d_norm = flat_1d/flat_per
    flat_flat = flat_mf / flat_1d_norm[None,:]
    return flat_flat


# In[ ]:

#Convert flat_flat to binary for tracing
def make_flat_flat_bin(flat_flat):
    flat_flat_bin = flat_flat.copy()

    for i in range(flat_flat.shape[1]):
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
    return flat_flat_bin


# In[ ]:

def make_fibre_centroids(flat_flat_bin):
    out_array, fibres = label(flat_flat_bin, np.ones((3,3)))
    print 'Found', fibres,'fibres for centroiding'
    # n-=2 # fibres 252 and 253 are not good for HD1581 epoch 0 

    #create centroid array
    cols = out_array.shape[1]
    fibre_centroids = np.ones((fibres,cols))*np.nan
    for fibre in range(fibres):
        wRows, wCols = np.where(out_array==fibre+1)
        print fibre,
        for col in range(max(wCols)+1):
            fibre_centroids[fibre, col] = np.average(wRows[wCols==col])
    return fibre_centroids


# In[ ]:

def make_single_fibre_centroids(flat_flat_bin, group):
    out_array, fibres = label(flat_flat_bin, np.ones((3,3)))
    print 'Found', fibres,'fibres for centroiding'
    # n-=2 # fibres 252 and 253 are not good for HD1581 epoch 0 

    #create centroid array
    cols = out_array.shape[1]
    fibre_centroids = np.ones(cols)*np.nan
#     for fibre in range(fibres):
    wRows, wCols = np.where(out_array==group)
#     print fibre,
    for col in range(max(wCols)+1):
        fibre_centroids[fibre, col] = np.average(wRows[wCols==col])
    return fibre_centroids


# In[ ]:

#create polynomials from centroids
def make_fibrePolys(fibre_centroids):
    fibrePolys = np.ones((fibre_centroids.shape[0],6))*np.nan
    for y,fibre in enumerate(fibre_centroids):
        fibrePolys[y-1,:] = np.polyfit(range(fibre.shape[0]),fibre,5)
        if np.sum(np.isnan(fibrePolys[y-1,:]))>0:
            print 'Found nan in fibre number',y
            print 'Fibre values',fibre[np.isnan(fibre)]
            print
    return fibrePolys


# In[ ]:

#create tramlines from polynomials
def make_tramlines(fibre_centroids, fibrePolys):
    tramlines = (np.ones(fibre_centroids.shape)*np.nan)[:-1]
    thisRange = np.arange(fibre_centroids.shape[1])
    for i,thisPoly in enumerate(fibrePolys[1:]):
        tramlines[i] = fithOrder(thisPoly,thisRange)
    return tramlines


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
##Need to SUBTRACT the result of the gaussian fit 
###to make the 1st curve be like the second (i.e the traces be like the arc)


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

def extract(tramlines_shifted, data):
    extracted = np.ones(tramlines_shifted.shape)*np.nan
    for fibre in range(tramlines_shifted.shape[0]):
        extracted[fibre] = sum_extract(fibre,tramlines_shifted, data, 4)
    return extracted


# In[1]:

def gaussian(x, mu, sig, ):
    x = np.array(x)
    return np.exp(-np.power(x - mu, 2.) / 2 / np.power(sig, 2.))


def flexi_gaussian(x, mu, sig, power, a, d ):
    """
    a* np.exp(-np.power(np.abs((x - mu) * np.sqrt(2*np.log(2))/sig),power))+d
    """
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

#takes a filename and an array and writes the npy in folder\
def write_NPY(fileName, prefix, postfix, data, folder =""):
    if folder[:-1]!="/": folder += "/"
    outName = folder + prefix + "_" + fileName.split("/")[-1][:-5] + "_" + postfix
    np.save(outName, data)
    print outName, "saved"


# In[ ]:

#takes a filename and an array and writes the npy in folder\
def read_NPY(fileName, prefix, postfix, folder =""):
    if folder[:-1]!="/": folder += "/"
    outName = folder + prefix + "_" + fileName.split("/")[-1][:-5] + "_" + postfix + ".npy"
    data = np.load(outName)
    print outName, "read"
    return data


# In[1]:

#Creates a ploynomial, a model and errors given a objectArc and a lineTemplate
def make_poly_model_err(objectArc, lineListFileName):

    #the 1/2 range to fit the cross-correlation over
    halfCCRange = 15 
    
    #1D array of 0s length of pixels
    lineTemplate = np.zeros(objectArc.shape[0]) 

    #this is the ThXe emission line list. 2 cols: pixel, wl. Comes from an adjusted linelist.txt
    lineList = np.loadtxt(lineListFileName)
    
    # make the wavelengths with emissions from lineList=1, the rest 0.
    lineTemplate[lineList[:,0].astype(int)]=1
    
    #plots template and arc
#     import pylab as plt
#     plt.plot(lineTemplate * np.max(objectArc))
#     plt.plot(objectArc)
#     plt.show()
    
    
    #Cross correlate to get offset in the px direction
    CCCurve = np.correlate(objectArc, lineTemplate, mode='full')
    
    if np.sum(np.isnan(CCCurve))==0: #Check for NaNs
        #prepare x and y of the cross correlation curve
        ccX = np.arange(-halfCCRange,halfCCRange+1)
        ccY = CCCurve[int(CCCurve.shape[0]/2.)+1-halfCCRange:int(CCCurve.shape[0]/2.)+1+halfCCRange+1]

        #Find px value of the peak of the cross correlation
        ccMaxIdx = np.where(ccY==np.max(ccY))[0][0]
        thisShift = ccX[ccMaxIdx]
        print 'Shift between template and this fibre arc', thisShift

        #creates adjLineList with the px offset found in the CC step
        adjLineList = lineList.copy()
        adjLineList[:,0] += thisShift


        #this loop fits a flexigaussian on each emission peak to fine tune the exact px value
        for i, thisLineWl in enumerate(adjLineList): #retuns i=index, thisLineWl=[px value, wl] 
    #         print i,'- Searching for wl',thisLineWl[1],'in px',thisLineWl[0]

            #slice 5px on each side of the initial px location guess
            firstSliceX = np.arange(thisLineWl[0]-5,thisLineWl[0]+6).astype(int)
            firstSliceY = objectArc[firstSliceX]

            #find the px value of the found peak
            maxIdx =  firstSliceX[np.where(firstSliceY==np.max(firstSliceY))[0][0]]

            #slice again, now around the found peak
            secondSliceX = np.arange(maxIdx-5,maxIdx+6).astype(int)
            secondSliceY = objectArc[secondSliceX]      

            p,_ = fit_flexi_gaussian([maxIdx,1., 2., np.max(secondSliceY), 0], secondSliceY, secondSliceX )

    #         print 'Changing pixel value',adjLineList[i,0], 'into', p[0]

            #put the found peak in the final array
            goodPxValue = p[0]
            adjLineList[i,0] = goodPxValue

            x_dense = np.linspace(np.min(secondSliceX),np.max(secondSliceX))
            plt.plot(secondSliceX,secondSliceY)
            plt.plot(x_dense,flexi_gaussian(x_dense,p[0],p[1],p[2],p[3],p[4]))
            plt.title(goodPxValue)
            plt.show()

    #     print adjLineList
    #         print

        #At this point we have an adjusted line list (adjLineList) with the best pixel values 
        #We turn that into a polynomial, a Wl solution, and an error array

        #create the polynimial solution from adjLineList
        thisPoly = np.polyfit(adjLineList[:,0], adjLineList[:,1], 5)

        #Evaluate the polynomial accross all pixels
        thisSolution = fithOrder(thisPoly, np.arange(4095))

        thisErr = fithOrder(thisPoly, adjLineList[:,0]) - adjLineList[:,1]

    else: #NaNs in CCCurve (from extracted_arc)
        thisPoly = np.ones(6) * np.nan
        thisSolution = np.ones(4095) * np.nan
        thisErr = np.ones(adjLineList.shape[0]) * np.nan

    return thisPoly, thisSolution, thisErr


# In[5]:

#Creates a ploynomial from a wls
# wlsfilename can be npy or txt
def make_poly_from_wls(wlsFileName):
    
    if wlsFileName[-3:]=="npy": #numpy array
        wls = np.load(wlsFileName)
    elif wlsFileName[-3:]=="txt": #txt file
        wls = np.loadtxt(wlsFileName)
            
    #create the polynimial solution from adjLineList
    thisPoly = np.polyfit(np.arange(wls.shape[0]), wls, 3)

    return thisPoly


# In[ ]:

#extend line list to all peaks
#creates the new linelist. Wrong, but equally wrong

def extend_lineList(objectArc, thisWlPoly, noiseLevel = 200):
    
    thisObjectArc = objectArc.copy()
    thisObjectArc[thisObjectArc<noiseLevel]=0
    thisPeaks = find_peaks_cwt(thisObjectArc, np.arange(1,2))
    print "Found", len(thisPeaks), "peaks."


    lineList_v2 = []
    for i, thisPeak in enumerate(thisPeaks):
    #         print 'Searching for wl',thisLineWl[1],'in px',thisLineWl[0],

        firstSliceX = np.arange(thisPeak-5,thisPeak+6).astype(int)
        if np.max(firstSliceX) < objectArc.shape[0]:
            firstSliceY = thisObjectArc[firstSliceX]
            maxIdx =  firstSliceX[np.where(firstSliceY==np.max(firstSliceY))[0][0]]

            secondSliceX = np.arange(maxIdx-5,maxIdx+6).astype(int)
            secondSliceY = thisObjectArc[secondSliceX]      

            p,_ = fit_flexi_gaussian([maxIdx,1., 2., np.max(secondSliceY), 0], secondSliceY, secondSliceX )

        #         print 'Changing pixel value',adjLineList[i,0], 'into', p[0]
            goodPxValue = p[0]

#             x_dense = np.linspace(np.min(secondSliceX),np.max(secondSliceX))
#             plt.plot(secondSliceX,secondSliceY)
#             plt.plot(x_dense,flexi_gaussian(x_dense,p[0],p[1],p[2],p[3],p[4]))
#             plt.title(goodPxValue)
#             plt.show()

            x = np.polynomial.polynomial.polyval(goodPxValue, thisWlPoly[::-1])

            lineList_v2.append((goodPxValue,x))

    lineList_v2 = np.array(lineList_v2)
    #         plt.plot(x_dense,gaussian(x_dense,p[0],p[1]))
    #         plt.plot(firstSliceX,firstSliceY/np.max(firstSliceY))
    #         plt.title(maxIdx)
    #         plt.plot(objectArc[thisLineWl[0]-5:thisLineWl[0]+5])
    #         plt.show()
    return lineList_v2
    


# In[3]:

#reduces the extendedLinelist based on line stability across epochs
def px_change_across_epochs(extendedLinelist, arcNpyList, goodFibres, folder):
    #sanitise folder
    if folder[:-1]!="/": folder += "/"
        
    #initialise offset array with nans. rows = number of lines, cols= epochs
    #each cell holds de diff in pixels between the first epoch and each subsequent for each line
    pxValuesEpoch = np.ones((extendedLinelist.shape[0],len(arcNpyList) )) * np.nan
    

    ref_arc = np.load(folder+arcNpyList[0]) 
    halfCCRange = 15
    refFibre = goodFibres[0]
    
    #loop over each arc of each epoch (columns)
    for colIdx in range(len(arcNpyList)):
        extractedFibre = goodFibres[colIdx]
#         print "Working with", arcNpyList[colIdx] 
        thisArcNpy = arcNpyList[colIdx]
        extracted_arc = np.load(folder+thisArcNpy)

#         thisWlsNpy = wlsNpyList[0]
#         print thisWlsNpy
    # bigLineList = np.loadtxt('bigLineList.txt')
#     bigLineList = reducedbigLineList
    # lineLocations = np.ones((extracted_arc.shape[0],bigLineList.shape[0]))*np.nan
#     lineLocations = np.ones(bigLineList.shape[0])*np.nan

# for thisFibre in range(extracted_arc.shape[0])[170:171]:
        print 'Fibre',refFibre,"epoch",colIdx
    
        #Cross correlate to get offset in the px direction
        CCCurve = np.correlate(ref_arc[refFibre], extracted_arc[extractedFibre], mode='full')

#         if np.sum(np.isnan(CCCurve))==0: #Check for NaNs
        #prepare x and y of the cross correlation curve
        ccX = np.arange(-halfCCRange,halfCCRange+1)
        ccY = CCCurve[int(CCCurve.shape[0]/2.)-halfCCRange:int(CCCurve.shape[0]/2.)+1+halfCCRange]

        #Find px value of the peak of the cross correlation
        ccMaxIdx = np.where(ccY==np.max(ccY))[0][0]
        thisShift = ccX[ccMaxIdx]
        print 'Shift between template and this fibre arc', thisShift

        #creates adjLineList with the px offset found in the CC step
        adjLineList_v2 = extendedLinelist.copy()
        adjLineList_v2[:,0] += thisShift

#         plt.plot(ccX,ccY)
#         plt.show()

#         thisArc = extracted_arc[refFibre].copy()
        
        
        #hack, should be refFibre for complete wlSolutions array (only doing 1 now)
#         thisWlSolution = wlSolutions[0].copy() 
    
        #for this epoch, loop over each wl found in the long list (v2)
        for rowIdx, thisWlpx in enumerate(adjLineList_v2[:,0]):
#             print 'Searching for wl',thisWl
#             diffArray = np.abs(thisWlSolution-thisWl)
#             print thisWl, thisWlSolution, diffArray 
#             wlPx = np.where(diffArray==np.min(diffArray))[0][0]
#             print wlPx, thisWlSolution[wlPx-1:wlPx+2]

            thisSlice = np.arange(thisWlpx-6,thisWlpx+7).astype(int)

#             firstSliceX = thisWlSolution[thisSlice]
            firstSliceX = thisSlice
            firstSliceY = extracted_arc[extractedFibre][thisSlice]
            maxIdx =  thisSlice[np.where(firstSliceY==np.max(firstSliceY))[0][0]]
#             print maxIdx
        
#             if rowIdx==0:
#                 if colIdx==0:
#                     plt.plot(firstSliceX,firstSliceY)
#                     plt.show()

        
            thisSecondSlice = np.arange(maxIdx-5,maxIdx+6).astype(int)
#             secondSliceX = thisWlSolution[thisSecondSlice]
            secondSliceX = thisSecondSlice
            secondSliceY = extracted_arc[extractedFibre][thisSecondSlice]      

#             if rowIdx>30:
#                 if colIdx==1:                    
#                     x_dense = np.linspace(np.min(secondSliceX),np.max(secondSliceX))
#                     plt.plot(secondSliceX,secondSliceY)
#                     plt.plot(x_dense,flexi_gaussian(x_dense,maxIdx,4., 2.1,np.max(secondSliceY),0))
#                     plt.show()

            p,_ = fit_flexi_gaussian([maxIdx,4., 2.1, np.max(secondSliceY), 0], secondSliceY, secondSliceX )
#             if rowIdx>30:
#                 if colIdx==1:
#                     x_dense = np.linspace(np.min(secondSliceX),np.max(secondSliceX))
#                     plt.plot(secondSliceX,secondSliceY)
#                     plt.plot(x_dense,flexi_gaussian(x_dense,p[0],p[1],p[2],p[3],p[4]))
#                     tit='line',rowIdx,'epoch',colIdx
#                     plt.title(tit)
#                     plt.show()

#             print secondSliceX, secondSliceY
#             print [maxIdx,.2, 2.8, np.max(secondSliceY), 2]
#             print 'F',p[0], extendedLinelist[rowIdx,0]
#             diff =np.abs(p[0]-extendedLinelist[rowIdx,0])
#             if diff>1.5:
#                 x_dense = np.linspace(np.min(secondSliceX),np.max(secondSliceX))
#                 plt.plot(secondSliceX,secondSliceY)
#                 plt.plot(x_dense,flexi_gaussian(x_dense,p[0],p[1],p[2],p[3],p[4]))
#                 tit='line',rowIdx,'epoch',colIdx
#                 plt.title(tit)
#                 plt.show()

            pxValuesEpoch[rowIdx, colIdx] = p[0]
    
    
    return pxValuesEpoch
#         lineLocations[thisFibre,i] = goodWlValue
#             lineLocations[i] = goodWlValue

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

def make_WLS_from_polys(refWLS, epochPolyFit):
    
    newWLS = []
    newPXs = []
    refPx = np.arange(refWLS.shape[0])
    for thisEpochPolyFit in epochPolyFit:
        
        thisNewPx = refPx + np.polynomial.polynomial.polyval(refPx, thisEpochPolyFit[::-1])
        newPXs.append(thisNewPx)

        thisNewWLS = np.interp(refPx, thisNewPx, refWLS)
        newWLS.append(thisNewWLS)
    return newPXs, newWLS


# In[6]:

def write_flux_and_wls(thisExtracted_obj, thisExtracted_arc, wls, epochIdx, fileIdx, folder):
    objFilename = folder + '/HD1581_' + str(epochIdx) + '_' + str(fileIdx) + '.txt'
    arcFilename = folder + '/ThXe_' + str(epochIdx) + '_' + str(fileIdx) + '.txt'
    np.savetxt(objFilename, np.vstack((wls[epochIdx], thisExtracted_obj)).transpose())
    print objFilename, 'saved.'
    np.savetxt(arcFilename, np.vstack((wls[epochIdx], thisExtracted_arc)).transpose())    
    print arcFilename, 'saved.'


# In[1]:

def robust_polyfit(x, y, order, sigmaClips, booPlot=False):

    for idx,i in enumerate(sigmaClips):
        a = x.copy()
        b = y.copy()
        
        poly = np.polyfit(a, b, order)
        res = np.polynomial.polynomial.polyval(a, poly[::-1])-b
        stdRes = np.std(res)
        
        polyFilter = np.abs(res)<=stdRes*i
        
        x = a[polyFilter].copy()
        y = b[polyFilter].copy()
        print 'Clipping (std*sigma)', stdRes*i,'.', x.shape, 'points left.'

        if booPlot==True:
            plt.plot(x,y,'.')
            if idx==0:
                ylim = plt.ylim()
            else:
                plt.ylim(ylim)
            plt.plot(x,np.polynomial.polynomial.polyval(x, poly[::-1]))
            plt.title(str(stdRes*i))
            plt.show()
        
    poly = np.polyfit(x, y, order)
    
    if booPlot==True:
        plt.plot(x,y,'.')
        plt.ylim(ylim)
        plt.plot(x,np.polynomial.polynomial.polyval(x, poly[::-1]))
        plt.title(str(stdRes*i))
        plt.show()
        
    return poly


# In[ ]:



