
# coding: utf-8

# In[1]:

# import pyfits as pf
# import pylab as plt
# from scipy import optimize
# from scipy.signal import medfilt, find_peaks_cwt
# from scipy.ndimage.filters import minimum_filter, maximum_filter, median_filter, convolve
# from scipy.ndimage.measurements import label
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
extracted_arc1 = MHT.read_NPY(arcFileName, "arc", "px", folder)
extracted_obj1 = MHT.read_NPY(objFileName, "obj", "px", folder)

flatFileName = '1_21aug/1/21aug10047.fits'
arcFileName = '1_21aug/1/21aug10046.fits'
objFileName = '1_21aug/1/21aug10041.fits'
extracted_arc2 = MHT.read_NPY(arcFileName, "arc", "px", folder)
extracted_obj2 = MHT.read_NPY(objFileName, "obj", "px", folder)

flatFileName = '2_22aug/1/22aug10032.fits'
arcFileName = '2_22aug/1/22aug10031.fits'
objFileName = '2_22aug/1/22aug10036.fits'
extracted_arc3 = MHT.read_NPY(arcFileName, "arc", "px", folder)
extracted_obj3 = MHT.read_NPY(objFileName, "obj", "px", folder)

flatFileName = '3_24aug/1/24aug10053.fits'
arcFileName = '3_24aug/1/24aug10054.fits'
objFileName = '3_24aug/1/24aug10058.fits'
extracted_arc4 = MHT.read_NPY(arcFileName, "arc", "px", folder)
extracted_obj4 = MHT.read_NPY(objFileName, "obj", "px", folder)

flatFileName = '4_25aug/1/25aug10039.fits'
arcFileName = '4_25aug/1/25aug10043.fits'
objFileName = '4_25aug/1/25aug10044.fits'
extracted_arc5 = MHT.read_NPY(arcFileName, "arc", "px", folder)
extracted_obj5 = MHT.read_NPY(objFileName, "obj", "px", folder)


# In[11]:

import pylab as plt
# for i in range(165, 175):
plt.plot(extracted_obj1[169], label = str(i))
plt.plot(extracted_obj2[169], label = str(i))
plt.plot(extracted_obj3[169], label = str(i))
plt.plot(extracted_obj4[169], label = str(i))
plt.plot(extracted_obj5[170], label = str(i))
# plt.legend()
plt.show()


# In[14]:

# import pylab as plt
# plt.plot(extracted_arc1[169])
# plt.plot(extracted_arc2[169])
# plt.plot(extracted_arc3[169])
# plt.plot(extracted_arc4[169])
# plt.plot(extracted_arc5[169])
# plt.plot(extracted_obj1[169])
# plt.plot(extracted_obj2[169])
# plt.plot(extracted_obj3[169])
# plt.plot(extracted_obj4[169])
# plt.plot(extracted_obj5[169])
# plt.show()


# In[58]:

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
    lineListfFileName = '../linelist_blue_v3.txt'
        
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


# In[60]:

import glob
import numpy as np

lineList_v3 = np.loadtxt("../lineList_blue_v3.txt")
reload(MHT)
pxValuesEpoch = MHT.px_change_across_epochs(
    lineList_v3, glob.glob1("herpy_out","WLS*"), 
    glob.glob1("herpy_out","arc*"), 169, folder)


# In[74]:

order = 3
epochPolyFit = []
epochDiffToFit = []
#the offsets of 1 epoch per loop
for i in range(pxValuesEpoch.shape[1]):
    
    polyFilter = [(pxValuesEpoch[:,i]-pxValuesEpoch[:,0])<100]
    
    thisEpochPolyFit = np.polyfit(pxValuesEpoch[:,0][polyFilter],(pxValuesEpoch[:,i]-pxValuesEpoch[:,0])[polyFilter], order)
    thisEpochDiffToFit = np.polynomial.polynomial.polyval(pxValuesEpoch[:,0], thisEpochPolyFit[::-1])-(pxValuesEpoch[:,i]-pxValuesEpoch[:,0])

    epochPolyFit.append(thisEpochPolyFit)
    epochDiffToFit.append(thisEpochDiffToFit)
#     plt.plot(pxValuesEpoch[:,0],pxValuesEpoch[:,i]-pxValuesEpoch[:,0], '.')
#     plt.plot(pxValuesEpoch[:,0],np.polynomial.polynomial.polyval(pxValuesEpoch[:,0], thisEpochPolyFit[::-1]))
#     plt.plot(pxValuesEpoch[:,0],thisEpochDiffToFit, '.')
#     plt.show()

#     print thisEpochDiffToFit





# In[28]:

MHT.write_NPY(arcFileName, "pol_f", "", wlPolys, folder =folder)
MHT.write_NPY(arcFileName, "WLS_f", "", wlSolutions, folder =folder)
MHT.write_NPY(arcFileName, "WLERR_f", "", wlErrors, folder =folder)


# In[ ]:




# In[84]:




# In[86]:

_,WLS_f = make_WLS_from_polys(wlSolutions[0], epochPolyFit)


# In[99]:




# In[100]:

write_flux_and_wls(extracted_arc1[170],extracted_arc1[170],WLS_f,0)
write_flux_and_wls(extracted_arc2[170],extracted_arc2[170],WLS_f,1)
write_flux_and_wls(extracted_arc3[170],extracted_arc3[170],WLS_f,2)
write_flux_and_wls(extracted_arc4[170],extracted_arc4[170],WLS_f,3)
write_flux_and_wls(extracted_arc5[170],extracted_arc5[170],WLS_f,4)


# In[97]:

plt.plot(WLS_f[0], extracted_arc1[170])
plt.plot(WLS_f[1], extracted_arc2[170])
plt.plot(WLS_f[2], extracted_arc3[170])
plt.plot(WLS_f[3], extracted_arc4[170])
plt.plot(WLS_f[4], extracted_arc5[170])
plt.show()


# In[96]:

plt.plot( extracted_arc1[170])
plt.plot(extracted_arc2[170])
plt.plot( extracted_arc3[170])
plt.plot(extracted_arc4[170])
plt.plot( extracted_arc5[170])
plt.show()


# In[87]:

WLS_f


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


# In[35]:

a = np.load("herpy_out/obj_20aug10053_px.npy")


# In[36]:

import pylab as plt
# plt.plot(np.nansum(a, axis=1))
# plt.show()


# In[39]:

plt.plot(wlSolutions[0],a[169])
plt.show()


# In[ ]:

a[170]


# In[32]:

wlSolutions[0]


# In[40]:

import glob


# In[45]:

wlnames = glob.glob1(folder,"WLS_f*")
objnames = glob.glob1(folder,"obj*")


# In[52]:

for i in range(5):
    print i 
    a = np.load(folder + "/" + wlnames[i])
    b = np.load(folder + "/" + objnames[i])
    c = b[169]
    print a[0]
#     plt.plot(a[0],c)
#     plt.plot(c)

# plt.show()


# In[ ]:



