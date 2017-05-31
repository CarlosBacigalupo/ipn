
# coding: utf-8

# In[12]:

import numpy as np

import myHerpyTools as MHT
reload(MHT)


# In[2]:

cd /Users/Carlos/Documents/HERMES/reductions/new_start_6.5/HD1581/


# In[3]:

folder = "herpy_out"


# In[4]:

flatFileName = '0_20aug/1/20aug10034.fits'
arcFileName = '0_20aug/1/20aug10052.fits'
objFileName = '0_20aug/1/20aug10053.fits'

# flatFileName = '1_21aug/1/21aug10047.fits'
# arcFileName = '1_21aug/1/21aug10046.fits'
# objFileName = '1_21aug/1/21aug10041.fits'

# flatFileName = '2_22aug/1/22aug10032.fits'
# arcFileName = '2_22aug/1/22aug10031.fits'
# objFileName = '2_22aug/1/22aug10036.fits'

# flatFileName = '3_24aug/1/24aug10053.fits'
# arcFileName = '3_24aug/1/24aug10054.fits'
# objFileName = '3_24aug/1/24aug10058.fits'

# flatFileName = '4_25aug/1/25aug10039.fits'
# arcFileName = '4_25aug/1/25aug10043.fits'
# objFileName = '4_25aug/1/25aug10044.fits'


# In[5]:

extracted_arc = MHT.read_NPY(arcFileName, "arc", "px", folder)


# In[6]:

#FULL LOOP

#Initialise the 3 output arrays
wlSolutions = []
wlErrors = []
wlPolys = []

#loop over each fibre
for thisFibre in range(extracted_arc.shape[0])[169:170]:
    print 'Fibre',thisFibre

    # get the flux from a single arc
    objectArc = extracted_arc[thisFibre].copy() 
    
    # Template to build model from.
    lineListfFileName = '../linelist_blue.txt'
        
    #Create the model, etc
    thisPoly, thisSolution, thisErr = MHT.make_poly_model_err(objectArc, lineListfFileName)
    
    #append lists
    wlPolys.append(thisPoly)
    wlSolutions.append(thisSolution)
    wlErrors.append(thisErr)

#turn lists into np arrays
wlPolys = np.array(wlPolys)
wlSolutions = np.array(wlSolutions)
wlErrors = np.array(wlErrors)


#         plt.plot(x_dense,gaussian(x_dense,p[0],p[1]))
#         plt.plot(firstSliceX,firstSliceY/np.max(firstSliceY))
#         plt.title(maxIdx)
# #         plt.plot(masterArc[thisLineWl[0]-5:thisLineWl[0]+5])
#         plt.show()


# In[ ]:

# MHT.write_NPY(arcFileName, "pol", "", wlPolys, folder =folder)
# MHT.write_NPY(arcFileName, "WLS", "", wlSolutions, folder =folder)
# MHT.write_NPY(arcFileName, "WLERR", "", wlErrors, folder =folder)


# In[13]:

#extend line list to all peaks
#creates the new linelist. Wrong, but equally wrong
lineList_v2 = MHT.extend_lineList(extracted_arc[169],wlPolys[0])


# In[14]:

import pylab as plt
plt.plot(wlSolutions[0], extracted_arc[169])
plt.scatter(lineList_v2[:,1],extracted_arc[169][lineList_v2[:,0].astype(int)])
plt.show()


# In[8]:

np.savetxt('../lineList_blue_v2.txt',lineList_v2)


# In[ ]:

# plt.plot(MHT.flexi_gaussian(range(11), 5,4, 1.5,30,0))
# plt.show()


# In[20]:

#Get the differences between the wlsolution for epoch 0 and the rest to identify unstable lines
import glob
reload(MHT)
pxValuesEpoch = MHT.px_change_across_epochs(
    lineList_v2, glob.glob1("herpy_out","WLS*"), 
    glob.glob1("herpy_out","arc*"), 169, folder)


# In[22]:

pxValuesEpoch


# In[26]:

plt.plot(pxValuesEpoch[:,0],pxValuesEpoch[:,0]-pxValuesEpoch[:,0], ".")
plt.plot(pxValuesEpoch[:,0],pxValuesEpoch[:,1]-pxValuesEpoch[:,0], ".")
plt.plot(pxValuesEpoch[:,0],pxValuesEpoch[:,2]-pxValuesEpoch[:,0], ".")
plt.plot(pxValuesEpoch[:,0],pxValuesEpoch[:,3]-pxValuesEpoch[:,0], ".")
plt.plot(pxValuesEpoch[:,0],pxValuesEpoch[:,4]-pxValuesEpoch[:,0], ".")
plt.show()


# In[ ]:

order = 3
epochPolyFit = []
epochDiffToFit = []

#the offsets of 1 epoch per loop
for i in range(pxValuesEpoch.shape[1]):
    polyFilter = [(pxValuesEpoch[:,i]-pxValuesEpoch[:,0])<10]
    thisEpochPolyFit = np.polyfit(pxValuesEpoch[:,0][polyFilter],(pxValuesEpoch[:,i]-pxValuesEpoch[:,0])[polyFilter], order)
    thisEpochDiffToFit = np.polynomial.polynomial.polyval(pxValuesEpoch[:,0], thisEpochPolyFit[::-1])-(pxValuesEpoch[:,i]-pxValuesEpoch[:,0])

    epochPolyFit.append(thisEpochPolyFit)
    epochDiffToFit.append(thisEpochDiffToFit)
#     plt.plot(pxValuesEpoch[:,0],pxValuesEpoch[:,i]-pxValuesEpoch[:,0], '.')
#     plt.plot(pxValuesEpoch[:,0],np.polynomial.polynomial.polyval(pxValuesEpoch[:,0], thisEpochPolyFit[::-1]))
#     plt.plot(pxValuesEpoch[:,0],thisEpochDiffToFit, '.')
#     plt.show()

#     print thisEpochDiffToFit



# In[ ]:

fibreFilter = np.sum(np.abs(epochDiffToFit)>0.4, axis=0)
# print fibreFilter, -fibreFilter.astype(bool)
print lineList_v2[-fibreFilter.astype(bool)].shape

np.savetxt('../lineList_blue_v3.txt',lineList_v2[-fibreFilter.astype(bool)])

