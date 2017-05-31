
# coding: utf-8

# In[ ]:




# In[1]:

import myHerpyTools as MHT
import os
reload(MHT)


# In[2]:

cd /Users/Carlos/Documents/HERMES/reductions/new_start_6.5/HD1581/


# In[3]:

folder = 'herpy_out'


# In[4]:

try:
    os.makedirs(folder)
except:
    pass


# In[ ]:

flatFileName = '0_20aug/1/20aug10034.fits'
arcFileName = '0_20aug/1/20aug10052.fits'
objFileName = '0_20aug/1/20aug10053.fits'

flatFileName = '1_21aug/1/21aug10047.fits'
arcFileName = '1_21aug/1/21aug10046.fits'
objFileName = '1_21aug/1/21aug10041.fits'

flatFileName = '2_22aug/1/22aug10032.fits'
arcFileName = '2_22aug/1/22aug10031.fits'
objFileName = '2_22aug/1/22aug10036.fits'

flatFileName = '3_24aug/1/24aug10053.fits'
arcFileName = '3_24aug/1/24aug10054.fits'
objFileName = '3_24aug/1/24aug10058.fits'

flatFileName = '4_25aug/1/25aug10039.fits'
arcFileName = '4_25aug/1/25aug10043.fits'
objFileName = '4_25aug/1/25aug10044.fits'


# In[ ]:

#Opening files
flat = MHT.openFile(flatFileName)
arc =  MHT.openFile(arcFileName)
obj =  MHT.openFile(objFileName)


# In[ ]:

flat_flat = MHT.make_flat_flat(flat)


# In[ ]:

#Check results
# plt.imshow(flat_flat)
# plt.show()


# In[ ]:

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


# In[ ]:

flat_flat_bin = MHT.make_flat_flat_bin(flat_flat)


# In[ ]:

# #Check results
# plt.plot(flat_flat_bin[:,102])
# plt.show()


# In[ ]:

# plt.imshow(out_array)
# plt.show()


# In[ ]:

# np.max(out_array)


# In[ ]:

fibre_centroids = MHT.make_fibre_centroids(flat_flat_bin)


# In[ ]:

# fibre_centroids


# In[ ]:

# MHT.plt.imshow(MHT.np.isnan(fibre_centroids))
# MHT.plt.show()


# In[ ]:

# for i in range(397):
#     print np.sum(np.isnan(fibre_centroids[3,:]))


# In[ ]:

#line to remove 251 and 252 that have nans
# fibre_centroids = np.delete(fibre_centroids,251,0) #2 times for epoch0
# fibre_centroids = np.delete(fibre_centroids,372,0) 3 times for epoch1
# np.sum(np.isnan(fibre_centroids),1)


# In[ ]:

fibrePolys = MHT.make_fibrePolys(fibre_centroids)


# In[ ]:

tramlines = MHT.make_tramlines(fibre_centroids, fibrePolys)


# In[ ]:

#find vertical shift
shift = MHT.find_vertical_shift(flat, arc) #result to be subtracted to the tramlines (1st array in the CC...)
tramlines_shifted = tramlines - shift


# In[ ]:

shift


# In[ ]:

# #gaussian fit results
# plt.plot(x,y)
# # plt.plot(x_dense,gaussian(x_dense,p[0],p[1]))
# plt.show()


# In[ ]:

extracted_arc = MHT.extract(tramlines_shifted, arc)
MHT.write_NPY(arcFileName, "arc", "px", extracted_arc, folder)


# In[ ]:

extracted_obj = MHT.extract(tramlines_shifted, obj)
MHT.write_NPY(objFileName, "obj", "px", extracted_obj, folder)


# In[ ]:

# MHT.plt.plot(MHT.np.median(extracted_arc,axis=0))
# MHT.plt.show()


# In[ ]:

# MHT.plt.plot(extracted_arc[170])
# MHT.plt.show()

