
# coding: utf-8

# In[1]:

import pylab as plt


# In[2]:

import numpy as np


# In[3]:

import glob


# In[4]:

import myHerpyTools as MHT


# In[5]:

baseFolder = "/Users/Carlos/Documents/HERMES/reductions/new_start_6.5/rhoTuc"


# In[ ]:

ls


# In[6]:

cd /Users/Carlos/Documents/HERMES/reductions/new_start_6.5/rhoTuc/herpy_out/3/


# In[ ]:

#HD1581
cam (0-3)- epoch (0-4)
0-0 169
0-1 169
0-2 169
0-3 169
0-3 169

1-0 169
1-1 169
1-2 169
1-3 169
1-4 169

2-0 215
2-1 168
2-2 168
2-3 267
2-4 169

3-0 169
3-1 169
3-2 169
3-3 188
3-4 187



# In[ ]:

#HD285507
cam (0-3) - epoch (0-4)
0-0 219
0-1 219
0-2 219
0-3 219
0-4 219

1-0 262
1-1 232
1-2 245
1-3 255
1-4 253


2-0 215
2-1 168
2-2 168
2-3 267
2-4 169



3-0 218
3-1 220
3-2 220
3-3 218
3-4 218



# In[ ]:

#print obj
for thisFile in fList:
    if "1_obj" in thisFile:
        npy = np.load(thisFile)
        plt.plot(npy[219], label = thisFile)
plt.legend(loc=0)
plt.show()    


# In[ ]:

#print arcs
for thisFile in fList:
    if "arc" in thisFile:
        npy = np.load(thisFile)
        plt.plot(npy[38], label = thisFile)
plt.legend(loc=0)
plt.show()    


# In[ ]:

# fList [0,2,6,10, 16]  #arcs
# fList [1,3,4,5,7,8,9,11,12,13,14,15,17,18,19]
npy = np.load(fList[1])
for i,flux in enumerate(npy):
    if np.median(flux)>2000:
        plt.plot(flux, label = i)
plt.legend(loc=0)
plt.show()    


# In[ ]:

flist = glob.glob('H*.txt')


# In[ ]:

for i in flist:
    
    a = np.loadtxt(i)
#     b = np.loadtxt("HD1581_3_1.txt")
    plt.plot(a)
    plt.show()


# In[ ]:

plt.plot(a)
plt.show()


# #### Plots tramlines in front of image with bundle ranges (to find a given target)

# In[ ]:




# In[ ]:

# # bkgndFile = baseFolder + '/0_20aug/1/20aug10039.fits' #1st obj
# # bkgndFile = baseFolder + '/1_21aug/1/21aug10038.fits' #1st obj
# # bkgndFile = baseFolder + '/2_22aug/1/22aug10041.fits' #1st obj
# # bkgndFile = baseFolder + '/3_24aug/1/24aug10063.fits' #1st obj
# # bkgndFile = baseFolder + '/4_25aug/1/25aug10049.fits' #1st obj
# # bkgndFile = baseFolder + '/0_20aug/1/20aug10037.fits' #tram

# # bkgndFile = baseFolder + '/0_20aug/2/20aug20039.fits' #1st obj
# # bkgndFile = baseFolder + '/1_21aug/2/21aug20038.fits' #1st obj
# # bkgndFile = baseFolder + '/2_22aug/2/22aug20041.fits' #1st obj
# # bkgndFile = baseFolder + '/3_24aug/2/24aug20063.fits' #1st obj
# # bkgndFile = baseFolder + '/4_25aug/2/25aug20049.fits' #1st obj
# # bkgndFile = baseFolder + '/0_20aug/2/20aug20037.fits' #tram

# # bkgndFile = baseFolder + '/0_20aug/4/20aug40039.fits' #1st obj
# # bkgndFile = baseFolder + '/1_21aug/4/21aug40038.fits' #1st obj
# # bkgndFile = baseFolder + '/2_22aug/4/22aug40041.fits' #1st obj
# # bkgndFile = baseFolder + '/3_24aug/4/24aug40063.fits' #1st obj
# # bkgndFile = baseFolder + '/4_25aug/4/25aug40049.fits' #1st obj
# # bkgndFile = baseFolder + '/0_20aug/4/20aug40037.fits' #tram

# bkg = MHT.openFile(bkgndFile)
# folder = baseFolder + '/herpy_out/3'

# # tramFileName = '20aug10037.fits'
# # pre = str(0) + '_' + 'tlm_s1'
# # tramFileName = '21aug10036.fits'
# # pre = str(1) + '_' + 'tlm_s1'
# # tramFileName = '22aug10040.fits'
# # pre = str(2) + '_' + 'tlm_s1'
# # tramFileName = '24aug10066.fits'
# # pre = str(3) + '_' + 'tlm_s1'
# # tramFileName = '25aug10047.fits'
# # pre = str(4) + '_' + 'tlm_s1'

# # tramFileName = '20aug20037.fits'
# # pre = str(0) + '_' + 'tlm_s1'
# # tramFileName = '21aug20036.fits'
# # pre = str(1) + '_' + 'tlm_s1'
# # tramFileName = '22aug20040.fits'
# # pre = str(2) + '_' + 'tlm_s1'
# # tramFileName = '24aug20066.fits'
# # pre = str(3) + '_' + 'tlm_s1'
# # tramFileName = '25aug20047.fits'
# # pre = str(4) + '_' + 'tlm_s1'

# # tramFileName = '20aug40037.fits'
# # pre = str(0) + '_' + 'tlm_s1'
# # tramFileName = '21aug40036.fits'
# # pre = str(1) + '_' + 'tlm_s1'
# # tramFileName = '22aug40040.fits'
# # pre = str(2) + '_' + 'tlm_s1'
# # tramFileName = '24aug40066.fits'
# # pre = str(3) + '_' + 'tlm_s1'
# # tramFileName = '25aug40047.fits'
# # pre = str(4) + '_' + 'tlm_s1'

# # post = 'cam1'
# # post = 'cam2'
# post = 'cam4'
# tlm = MHT.read_NPY(tramFileName, pre, post, folder)


# In[7]:

fList = glob.glob("*.npy")
fList


# In[ ]:

#rhoTuc
cam (0-3) - epoch (0-4)
0-0 169 1-0 169 2-0 169 3-0 169
0-1 169 1-1 169 2-0 169 3-1 169
0-2 169 1-2 169 2-0 169 3-2 169
0-3 169 1-3 169 2-0 169 3-3 169
0-4 169 1-4 169 2-0 169 3-4 169
0-5 169 1-5 169 2-0 169 3-5 169
0-6 168 1-6 168 2-0 169 3-6 168
0-7 168 1-7 168 2-0 169 3-7 168


# In[22]:

#rhoTuc
# bkgndFile = baseFolder + '/0_20aug/1/20aug10044.fits' #1st obj
# bkgndFile = baseFolder + '/1_21aug/1/21aug10033.fits' #1st obj
# bkgndFile = baseFolder + '/2_21aug/1/21aug10044.fits' #1st obj
# bkgndFile = baseFolder + '/3_22aug/1/22aug10033.fits' #1st obj
# bkgndFile = baseFolder + '/4_22aug/1/22aug10046.fits' #1st obj
# bkgndFile = baseFolder + '/5_24aug/1/24aug10055.fits' #1st obj
# bkgndFile = baseFolder + '/6_25aug/1/25aug10040.fits' #1st obj
# bkgndFile = baseFolder + '/7_25aug/1/25aug10052.fits' #1st obj
# bkgndFile = baseFolder + '/0_20aug/1/20aug10037.fits' #tram

# bkgndFile = baseFolder + '/0_20aug/2/20aug20044.fits' #1st obj
# bkgndFile = baseFolder + '/1_21aug/2/21aug20033.fits' #1st obj
# bkgndFile = baseFolder + '/2_21aug/2/21aug20044.fits' #1st obj
# bkgndFile = baseFolder + '/3_22aug/2/22aug20033.fits' #1st obj
# bkgndFile = baseFolder + '/4_22aug/2/22aug20046.fits' #1st obj
# bkgndFile = baseFolder + '/5_24aug/2/24aug20055.fits' #1st obj
# bkgndFile = baseFolder + '/6_25aug/2/25aug20040.fits' #1st obj
# bkgndFile = baseFolder + '/7_25aug/2/25aug20052.fits' #1st obj

bkgndFile = baseFolder + '/0_20aug/4/20aug40044.fits' #1st obj
bkgndFile = baseFolder + '/1_21aug/4/21aug40033.fits' #1st obj
bkgndFile = baseFolder + '/2_21aug/4/21aug40044.fits' #1st obj
bkgndFile = baseFolder + '/3_22aug/4/22aug40033.fits' #1st obj
bkgndFile = baseFolder + '/4_22aug/4/22aug40046.fits' #1st obj
bkgndFile = baseFolder + '/5_24aug/4/24aug40055.fits' #1st obj
bkgndFile = baseFolder + '/6_25aug/4/25aug40040.fits' #1st obj
bkgndFile = baseFolder + '/7_25aug/4/25aug40052.fits' #1st obj

bkg = MHT.openFile(bkgndFile)
folder = baseFolder + '/herpy_out/3'

# tramFileName = '20aug10042.fits'
# pre = str(0) + '_' + 'tlm_s1'
# tramFileName = '21aug10031.fits'
# pre = str(1) + '_' + 'tlm_s1'
# tramFileName = '21aug10047.fits'
# pre = str(2) + '_' + 'tlm_s1'
# tramFileName = '22aug10032.fits'
# pre = str(3) + '_' + 'tlm_s1'
# tramFileName = '22aug10044.fits'
# pre = str(4) + '_' + 'tlm_s1'
# tramFileName = '24aug10053.fits'
# pre = str(5) + '_' + 'tlm_s1'
# tramFileName = '25aug10039.fits'
# pre = str(6) + '_' + 'tlm_s1'
# tramFileName = '25aug10053.fits'
# pre = str(7) + '_' + 'tlm_s1'

# tramFileName = '20aug20042.fits'
# pre = str(0) + '_' + 'tlm_s1'
# tramFileName = '21aug20031.fits'
# pre = str(1) + '_' + 'tlm_s1'
# tramFileName = '21aug20047.fits'
# pre = str(2) + '_' + 'tlm_s1'
# tramFileName = '22aug20032.fits'
# pre = str(3) + '_' + 'tlm_s1'
# tramFileName = '22aug20044.fits'
# pre = str(4) + '_' + 'tlm_s1'
# tramFileName = '24aug20053.fits'
# pre = str(5) + '_' + 'tlm_s1'
# tramFileName = '25aug20039.fits'
# pre = str(6) + '_' + 'tlm_s1'
# tramFileName = '25aug20053.fits'
# pre = str(7) + '_' + 'tlm_s1'

tramFileName = '20aug40042.fits'
pre = str(0) + '_' + 'tlm_s1'
tramFileName = '21aug40031.fits'
pre = str(1) + '_' + 'tlm_s1'
tramFileName = '21aug40047.fits'
pre = str(2) + '_' + 'tlm_s1'
tramFileName = '22aug40032.fits'
pre = str(3) + '_' + 'tlm_s1'
tramFileName = '22aug40044.fits'
pre = str(4) + '_' + 'tlm_s1'
tramFileName = '24aug40053.fits'
pre = str(5) + '_' + 'tlm_s1'
tramFileName = '25aug40039.fits'
pre = str(6) + '_' + 'tlm_s1'
tramFileName = '25aug40053.fits'
pre = str(7) + '_' + 'tlm_s1'

# post = 'cam1'
# post = 'cam2'
post = 'cam4'
tlm = MHT.read_NPY(tramFileName, pre, post, folder)


# In[23]:

#These go net to the bundles to identify them
bundleTicks = np.arange(0, bkg.shape[0], bkg.shape[0]/40.)+bkg.shape[0]/80.
a = np.arange(40)*10
b = np.arange(1,41)*10
bundleTitles = zip(a,b)

# fig, ax1 = plt.subplots()
# ax1.plot(df["..."])
# # ...
# ax2.plot(df["Market"])
# ax2.set_ylim([0, 5])

#set the image

# ax1.set_yticks(bundleTicks)# plt.yticks(bundleTicks, bundleTitles)
# ax1.set_yticklabels(bundleTitles)



# ax2 = ax1.twinx()

# ax2.yaxis = ax1.get_yaxis()
# ax2.set_yticks(tlm[:,0])# yticks(tramlines_shifted[:,0], range(tramlines_shifted.shape[0]))
# ax2.set_yticklabels(range(tlm.shape[0]))
# ax2 = ax1.twinx()

#plot the tramlines
for i in tlm:
    plt.plot(i)

plt.yticks(tlm[:,0], range(tlm.shape[0]))

rg = 50
cen = 1799.0
plt.ylim(cen-rg, cen+rg)
plt.xlim(10,120)
plt.imshow(np.log(bkg))


plt.show()


# In[ ]:

zip(bundleTicks, bundleTitles)


# In[ ]:

a = np.arange(40)*10
b = np.arange(1,41)*10
a.astype('str'), b.astype('str')
print zip(a,b)


# In[ ]:



