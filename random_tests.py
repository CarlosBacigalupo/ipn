
# coding: utf-8

# In[12]:

cd ~/Documents/HERMES/reductions/6.5/m67_lr/


# In[ ]:

i=0
for x,y,label in zip(thisCam.wavelengths, thisCam.red_fluxes, thisCam.fileNames):
    plt.plot(x,y+i, label= label, c='k')
    i+=300
plt.title(thisStar.name)
# plt.legend(loc = 0)
plt.show()


# In[ ]:

for i in a.sigmas[:]:
    plt.plot(i)
plt.show()


# In[ ]:

np.sum(thisCam.wavelengths,1)
np.sum(np.isnan(thisCam.wavelengths))


# In[ ]:

mid_px = thisCam.wavelengths.shape[1]/2
dWl = (thisCam.wavelengths[0,mid_px+1]-thisCam.wavelengths[0,mid_px]) / thisCam.wavelengths[0,mid_px]
RV = dWl * 0.5 * 3e8
print 'RV',RV, mid_px, thisCam.wavelengths[0,mid_px+1], thisCam.wavelengths[0,mid_px]


# In[2]:

from scipy.ndimage.measurements import label
import numpy as np


# In[57]:

from scipy import nanmedian


# In[7]:

a = np.random.rand(10)>0.5
print a,label(a)


# In[19]:

import pickle
import RVTools as RVT
import pylab as plt
# filename = 'HD1581.obj'
# filename = 'Brght01.obj'
# filename = 'red_Giant01.obj'
# filename = 'Giant01.obj'
filename = 'obj/HD1581.obj'
filename = 'obj/M67-375_216_1_56668.5921065.obj'
filename = 'obj/M67-381_186_0_56643.6659144.obj'
filename = 'obj/M67-381_186_0_56664.5809375.obj'

filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)
thisCam = thisStar.exposures.cameras[0]


# In[60]:

flux = thisCam.red_fluxes[0].copy()
flux2 = flux.copy()
flux[:30] = np.nan
flux[3336:3339] = np.nan
flux[4080:4097] = np.nan
plt.plot(flux)
plt.plot(flux2)
# plt.show()


# In[62]:

nanMap = np.isnan(flux)

if np.sum(nanMap)<flux.shape[0]:

    nanGroups, nNanGroups = label(nanMap)

    for thisGroup in range(1,nNanGroups+1):
        pxFix = np.where(nanGroups==thisGroup)[0]
#         print pxFix,
        if np.min(pxFix)==0: #is it at the beggining?
            print 'beginning'
            vlMin = nanmedian(flux)
            vlMax = flux[np.max(pxFix)+1]
        elif np.max(pxFix)==(flux.shape[0]-1):  #is it at the end?
            print 'end'
            vlMin = flux[np.min(pxFix)-1]
            vlMax = nanmedian(flux)
        else:
            print 'middle'
            vlMin = flux[np.min(pxFix)-1]
            vlMax = flux[np.max(pxFix)+1]
    
    print pxFix
    flux[pxFix] = np.linspace(vlMin, vlMax, num=pxFix.shape[0])

plt.plot(flux)
plt.show()

# #     leftEdgeIdx=0
# #     rightEdgeIdx=len(flux)

# #     plt.plot(nanMap)
# #     plt.show()

# #     nanMapIdx = np.where(nanMap==True) <<<<<make the next lines faster by using this
# if np.sum(nanMap)>0:
#     print 'Found NaNs in flux array'

# for i,booI in enumerate(nanMap):
#     if booI==False:
#         leftEdgeIdx = i
#         break

# for j,rbooI in enumerate(nanMap[::-1]):
#     if rbooI==False:
#         rightEdgeIdx = len(nanMap)-j
#         break        

# fluxMedian = stats.nanmedian(flux)
# if leftEdgeIdx>0:
#     flux[:leftEdgeIdx] = np.linspace(fluxMedian, flux[leftEdgeIdx+1],leftEdgeIdx)
# if rightEdgeIdx<len(flux):
#     flux[rightEdgeIdx:] = np.linspace(flux[rightEdgeIdx-1], fluxMedian, len(flux)-rightEdgeIdx)

# nanMap = np.isnan(flux)        
# if np.sum(nanMap)>0:
#     print 'NaNs remain in flux array'        

# plt.plot(nanMap)
# plt.show()


# In[ ]:

import pickle
import pylab as plt
import numpy as np


# In[ ]:

cd ~/Documents/HERMES/reductions/47Tuc_core_6.2/


# In[ ]:

print np.all([np.nansum(thisCam.red_fluxes,1).astype(bool) for thisCam in thisStar.exposures.cameras],0)


# In[ ]:

plt.plot(thisCam.red_fluxes[0])
plt.show()


# In[ ]:

print thisCam.SNRs
print np.nansum(thisCam.red_fluxes,1)
print thisCam.fileNames


# In[ ]:

thisStar.exposures.JDs.shape[0]


# In[ ]:

thisStar.exposures.JDs.shape


# In[ ]:

import scipy as sp


# In[ ]:

from scipy import optimize


# In[ ]:

thisStar.exposures.abs_baryVels


# In[ ]:

pwd


# In[ ]:

print RVs[0].shape


# In[ ]:

cd '/Users/Carlos/Documents/HERMES/reductions/47Tuc_core_6.2'


# In[ ]:

RVs = np.load('RVs.npy')
SNRs = np.load('SNRs.npy')


# In[ ]:

for epoch in range(RVs.shape[1]):
    cam = 0
    R = RVs[:,epoch,cam]
    S = SNRs[:,epoch,cam]
    a = np.histogram(R)
    plt.bar(a[1][:-1],a[0], width = (a[1][-2]-a[1][-1])*0.7)
    plt.plot(R,S,'.', c='r')
    plt.show()


# In[ ]:

a = np.histogram(R)
# plt.plot(a[1][:-1],a[0],'.')
plt.bar(a[1][:-1],a[0], width = (a[1][-2]-a[1][-1])*0.7)
plt.show()


# In[ ]:

a = np.histogram(R)
plt.plot(R,S,'.', c='r')
# plt.bar(a[1][:-1],a[0], width = (a[1][-2]-a[1][-1])*0.7)
plt.show()


# In[ ]:

R = RVs[:,15,0]
S = SNRs[:,15,0]


# In[ ]:

a = np.histogram(R)
# plt.plot(a[1][:-1],a[0],'.')
plt.bar(a[1][:-1],a[0], width = (a[1][-2]-a[1][-1])*0.7)
plt.show()


# In[ ]:

import os


# In[ ]:

os.curdir


# In[ ]:

pwd


# In[ ]:

os.getcwd().split('/')[-1]


# In[ ]:

plt.plot(thisCam.red_fluxes)
plt.show()


# In[ ]:

i=6
np.nanmean(thisCam.red_fluxes[i])/np.std(thisCam.red_fluxes[i])
print np.sqrt(np.nanmean(thisCam.red_fluxes[i]))
print np.nansum(thisCam.red_fluxes[i])
print stats.nanmedian(thisCam.red_fluxes[i])/stats.nanstd(thisCam.red_fluxes[i])


# In[ ]:

PyAstronomy


# In[ ]:

import PyAstronomy


# In[173]:

cd ~/Documents/HERMES/reductions/6.5/m67_lr/


# In[185]:

import pickle
# filename = 'HD1581.obj'
# filename = 'Brght01.obj'
# filename = 'red_Giant01.obj'
# filename = 'Giant01.obj'
filename = 'obj/HD1581.obj'
filename = 'obj/M67-375_216_1_56668.5921065.obj'
filename = 'obj/M67-381_186_0_56643.6659144.obj'
filename = 'obj/M67-381_186_0_56664.5809375.obj'

filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)
thisCam = thisStar.exposures.cameras[0]


# In[186]:

thisStar.exposures.MJDs


# In[187]:

import pylab as plt
plt.plot(thisCam.red_fluxes[0])
plt.show()


# In[ ]:

from PyAstronomy import pyasl

def baryTest2(baryVels, JDs):

    for i,jd in enumerate((JDs+2400000.5)[:]):
        heli, bary = pyasl.baryvel(jd, deq=2000.0)
#         print "Earth's velocity at JD: ", jd
#         print "Heliocentric velocity [km/s]: ", heli
#         print "Barycentric velocity [km/s] : ", bary

        # Coordinates of Sirius
        ra  = 101.28715535
        dec = -16.71611587
        
        #thisStar coords
        ra  = np.rad2deg(thisStar.RA)
        dec = np.rad2deg(thisStar.Dec)
        print np.rad2deg(thisStar.RA), np.rad2deg(thisStar.Dec), thisStar.name, thisStar.exposures.abs_baryVels[i]

        
        vh, vb = pyasl.baryCorr(jd, ra, dec, deq=2000.0)
        print "Barycentric velocity of Earth toward",thisStar.name,'[m/s]', vb*1000
        print vb*1000-thisStar.exposures.abs_baryVels[i]
        print ''


# In[ ]:

bary = np.load('npy/baryVels.npy')
print bary, thisStar.exposures.abs_baryVels`


# In[ ]:

from PyAstronomy import pyasl

def baryTest(thisStar):

    for i,jd in enumerate((thisStar.exposures.JDs+2400000.5)[:]):
        heli, bary = pyasl.baryvel(jd, deq=0)
#         print "Earth's velocity at JD: ", jd
#         print "Heliocentric velocity [km/s]: ", heli
#         print "Barycentric velocity [km/s] : ", bary

        # Coordinates of Sirius
        ra  = 101.28715535
        dec = -16.71611587
        
        #thisStar coords
        ra  = thisStar.RA
        dec = thisStar.Dec
        print thisStar.RA/15, thisStar.Dec, thisStar.name, thisStar.exposures.abs_baryVels[i]

        
        vh, vb = pyasl.baryCorr(jd, ra, dec, deq=0)
        print "Barycentric velocity of Earth toward",thisStar.name,'[m/s]', vb*1000
        print vb*1000-thisStar.exposures.abs_baryVels[i]
        print ''


# In[ ]:

from iraf import pyraf


# In[ ]:

import numpy as np
baryTest(thisStar)


# In[ ]:

barys = np.load('npy/baryVels.npy')


# In[ ]:

HD1581 coords
Right ascension	00h 20m 04.25995s
Declination	−64° 52′ 29.2549″


# In[ ]:

print thisStar.RA_dec, thisStar.Dec_dec#, thisStar.RA_h, thisStar.RA_min , thisStar.RA_sec


# In[ ]:

import toolbox
toolbox.dec2sex(thisStar.RA_dec)


# In[ ]:

import numpy as np


# In[ ]:

thisStar.RA_dec, toolbox.dec2sex(np.rad2deg(thisStar.RA_dec)/15)


# In[ ]:

00 20 06.49 -64 52 06.6


# In[ ]:

np.deg2rad(toolbox.sex2dec(0,20,06.49)*15)


# In[ ]:

np.deg2rad(-toolbox.sex2dec(64,52,6.6))


# In[ ]:

-toolbox.sex2dec(64,52,6.6)


# In[ ]:

RVs = np.random.random(100)
stdRV= np.std(RVs)
medRV = 0.5
sigmaClip = 0.1
print RVs,stdRV,medRV
print RVs[(RVs>=medRV-sigmaClip*stdRV) & (RVs<=medRV+sigmaClip*stdRV)]


# In[ ]:

import pylab as plt


# In[ ]:

g = plt.gca()
g.xaxis.majorTic
plt.show()


# In[ ]:

import matplotlib.pyplot as plt



# In[ ]:

fig = plt.figure()
# plt.title(title)

ax = fig.add_subplot(111)
# ax.bar(hist[1][1:],hist[0], width = (hist[1][-2]-hist[1][-1]))
# ax.grid()
# ax.set_ylabel('Counts')
# ax.set_xlabel('RV [m/s]')
# #         ax.set_ylim(0,10)

# ax2 = ax.twinx()
# ax2.scatter(R,S, c='r', s=100)
ax.bar(0,1 , width = (1), color='k')


# In[ ]:

plt.show()


# In[ ]:

pwd


# In[ ]:

cd 47Tuc_core_6.2/


# In[ ]:

cd obj


# In[ ]:

import pickle
# filename = 'HD1581.obj'
# filename = 'Brght01.obj'
# filename = 'red_Giant01.obj'
filename = 'red_Brght01.obj'
# filename = 'Giant01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)
thisCam = thisStar.exposures.cameras[3]


# In[ ]:

print thisStar.exposures
thisCam.fileNames


# In[ ]:

thisCam.red_fluxes


# In[ ]:

float(np.sum(np.isnan(SNRs)))/(SNRs.shape[0]*SNRs.shape[1]*SNRs.shape[2])*100


# In[ ]:

import pylab as plt
plt.plot(RVs[:,:,0])
plt.show()


# In[ ]:

data=np.load('npy/data.npy')
RVs=np.load('npy/RVs.npy')
#     sigmas=np.load('npy/sigmas.npy')
#     baryVels=np.load('npy/baryVels.npy')
#     JDs=np.load('npy/JDs.npy')
SNRs = np.load('npy/SNRs.npy')


# In[ ]:

pwd


# In[ ]:


filehandler = open('obj/red_N104-S1084.obj', 'r')
thisStar = pickle.load(filehandler)


# In[ ]:

thisCam = thisStar.exposures.cameras[0]


# # Bary tests

# In[ ]:

import toolbox as tb
from PyAstronomy import pyasl



# In[ ]:

MJD = 57131.84792 
# RA = tb.deg2rad(tb.sex2dec(0,24,05.67)*15) #47tuc in rad
# Dec = tb.deg2rad(-tb.sex2dec(72,4,52.6)) #47Tuc in rad
RA = tb.deg2rad(tb.sex2dec(0,20,06.49)*15) #HD1581 in rad
Dec = tb.deg2rad(-tb.sex2dec(64,52,06.6)) #HD1581 in rad


# In[ ]:

00 20 06.49 -64 52 06.6


# In[ ]:

MJDs = []
RVs = []
RV2s = []
RV3s = []
for i in np.arange(MJD-200, MJD+200):
    vh, vb = tb.baryvel(i+2400000+0.5) 
    vh2, vb2 = pyasl.baryvel(i+2400000+0.5, deq = 0.0)
    __, RV3 = pyasl.baryCorr(i+2400000+0.5,tb.rad2deg(RA), tb.rad2deg(Dec))
    
    RV = (vb[0]*np.cos(Dec)*np.cos(RA) + vb[1]*np.cos(Dec)*np.sin(RA) + vb[2]*np.sin(Dec))*1000
    RV2 = (vb2[0]*np.cos(Dec)*np.cos(RA) + vb2[1]*np.cos(Dec)*np.sin(RA) + vb2[2]*np.sin(Dec))*1000
    MJDs.append(i)
    RVs.append(RV)
    RV2s.append(RV2)
    RV3s.append(RV3*1000)
RVs = np.array(RVs)
RV2s = np.array(RV2s)
RV3s = np.array(RV3s)


# In[ ]:

plt.plot(MJDs, RVs)
plt.plot(MJDs, RV2s)
plt.plot(MJDs, RV3s)
plt.show()

plt.plot(MJDs, RVs-RV2s, label = '1-2')
plt.plot(MJDs, RV2s-RV3s, label = '2-3')
plt.legend(loc=0)
plt.show()



# # plot app

# In[ ]:

import numpy as np
import pylab as plt
a = np.random.rand(10)
b = np.random.rand(10)
c = np.random.rand(10)
plt.scatter(a,b)
plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('custom picker for line data')
line, = ax1.plot(a, b, 'o', picker=5)
fig.canvas.mpl_connect('pick_event', onpick2)


def onpick2():
    a=2
    print a


# In[ ]:

def aaa():
    xxx = 'sdd'
    print 'asdasd'
    return xxx


# In[ ]:

def test():
    print 'adasd'


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

import glob
import numpy as np
import os

os.chdir('/Users/Carlos/Documents/HERMES/reductions/')
a = glob.glob('*')
for i in a:
    if i!='HD1581_6.0':
        print i,
        b = np.load(i+'/npy/data.npy')
        print b.shape[0]


# In[ ]:

data


# In[ ]:

cd HD285507_1arc_6.2/


# In[ ]:

cd HERMES/reductions/HD285507_1arc_6.2/


# In[ ]:

import glob
import numpy as np
import os
data=np.load('npy/data.npy')
RVs=np.load('npy/RVs.npy')
SNRs=np.load('npy/SNRs.npy')


# In[ ]:

RVs.shape
W = np.zeros(np.hstack((RVs.shape, RVs.shape[0])))
for thisStarIdx in range(RVs.shape[0]):
    W1 = np.ones(RVs.shape)/(RVs.shape[0]-1)
    W1[thisStarIdx,:,:]=0
    W[:,:,:,thisStarIdx]=W1
    


# In[ ]:

data[thisStarIdx,2].astype(float).astype(int)


# In[ ]:

import RVTools as RVT
reload(RVT)
import pylab as plt
import pandas as pd


# In[ ]:

data=np.load('npy/data.npy')
RVs=np.load('npy/RVs.npy')
SNRs=np.load('npy/SNRs.npy')


# In[ ]:

# SNRs[:,:,0][SNRs[:,:,0]<1]
SNRs[np.isnan(SNRs)]=0
SNRs+=1e-17
create_allW(data,SNRs)


# In[ ]:

import numpy as np
import pylab as plt


# In[ ]:

cd /Users/Carlos/Documents/HERMES/reductions/6.2/HD285507_1arc_6.2/


# In[ ]:

allW = np.load('npy/allW_DM.npy')
allW[0,:,0][0]


# In[ ]:


# deltay = np.linspace(0, 4000)
# deltay = np.linspace(-2000, 2000)
# deltay = np.linspace(-4000, 0)
# SNRs = np.ones(50)*30
# SNRs = np.linspace(10, 100)
# SNRs = np.linspace(100, 10)
# W = calibrator_weights2(deltay,SNRs)

data=np.load('npy/data.npy')
# RVs=np.load('npy/RVs.npy')
SNRs=np.load('npy/SNRs.npy')
allW = np.load('npy/allW_PM.npy')
idx = np.where(data[:,0]=='Giant01')[0]
for cam in range(4):
    W = allW[:,cam,idx]

    thisSNRs = SNRs[:,0,cam]

#     plt.plot(deltay/np.max(np.abs(deltay)), label = 'deltay')
    plt.plot(thisSNRs, label= 'SNR')
    plt.plot(W*np.nanmax(thisSNRs), label = 'W')
    plt.legend(loc=0)
    # title = 'PM - deltay '+str(np.min(deltay))+','+str(np.max(deltay))+' - SNR '+str(np.min(SNRs))+','+str(np.max(SNRs))
#     title = 'DM - deltay '+str(np.min(deltay))+','+str(np.max(deltay))+' - SNR '+str(np.min(SNRs))+','+str(np.max(SNRs))
    title = 'PM - cam '+str(cam)+ ' ,' + str(data[idx])
    plt.title(title)
    plt.grid(True)
    plt.savefig(('PM_'+str(cam)))
    plt.show()


# In[ ]:

import numpy as np


# In[ ]:

def create_allW(data = [], SNRs = []):

    if ((data!=[]) and (SNRs!=[])):

        #load function that translates pivot# to y-pixel  p2y(pivot)=y-pixel of pivot
        p2y = RVT.pivot_to_y('/Users/Carlos/Documents/HERMES/reductions/6.2//rhoTuc_6.2/0_20aug/1/20aug10042tlm.fits') 

        #gets the y position of for the data array
        datay = p2y[data[:,2].astype(float).astype(int)]

        #Creates empty array for relative weights
        #allW[Weights, camera, staridx of the star to be corrected]
        allW = np.zeros((data.shape[0],4,data.shape[0]))

        for thisStarIdx in range(data.shape[0]):

            #converts datay into deltay
            deltay = datay-datay[thisStarIdx]
            for cam in range(2):

                thisSNRs = SNRs[:,0,cam].copy()
                thisSNRs[np.isnan(thisSNRs)]=1  #sets NaNs into SNR=1 

                W = calibrator_weights(deltay,thisSNRs)
                allW[:,cam,thisStarIdx] = W

                order = np.argsort(deltay)
                plt.plot(deltay[order], label = 'deltay')
                plt.plot(thisSNRs[order], label= 'SNR')
                plt.plot((W*np.max(deltay))[order], label = 'W')
                plt.legend(loc=0)
                plt.show()

        # a= pd.DataFrame(deltay)
        # # a.columns = labels
        # print a.head(n=40)
        # print W, np.sum(W)
    else:
        print 'Create allW: Input arrays missing'
        allW =[]

    return allW


# In[ ]:

def create_RVCorr(RVs, allW, RVClip = 1e17):
    RVCorr = np.zeros(RVs.shape)
    RVs[np.abs(RVs)>RVClip]=0
    for thisStarIdx in range(data.shape[0]):
        for epoch in range(RVs.shape[1]):
            for cam in range(4):
                RVCorr[thisStarIdx,epoch,cam] = np.nansum(allW[:,cam,thisStarIdx]*RVs[:,epoch,cam])
    return RVCorr


# In[ ]:

# for i in range(40):
i=37
plt.plot(RVs[i,:,0], label = 'RV')
plt.plot(RV_corr[i,:,0], label = 'Correction')
plt.plot(RVs[i,:,0]-RV_corr[i,:,0], label = 'Result', marker = 'o')
plt.legend(loc=0)
plt.show()


# In[ ]:

np.where(data[:,0]=='Giant01')


# In[ ]:

for epoch in range(RVs.shape[1]):
    print np.sum(RVs[:,epoch,1])/RVs.shape[1]


# In[ ]:

def create_corrRVs(RVs,W):
#Creates corrRVs with RV corrections for each RV. 
#has the same shape than RVs and W
#RVs-corrRVs = trueRVs (values without systematics)

    corrRVs = np.ones(RVs.shape)*np.nan
    
    #1 - loop retarded method. should be array operation.
    
    #check shape
    if ((RVs.shape==W.shape[:3]) and (len(W.shape)==4) and (RVs.shape[0]==W.shape[3])):
        for thisStaridx in range(RVs.shape[0]):
            for epoch in range(RVs.shape[1]):
                for cam in range(RVs.shape[2]):
                    corrRVs[thisStaridx,epoch,cam] = np.sum(RVs[:,epoch,cam]*W[:,epoch,cam,thisStaridx])
                
    else:
        print 'Bad array shape.'
        print 'RVs=', RVs.shape
        print 'W=', W.shape
        
    return corrRVs


# ### Solar Spectrum

# In[ ]:

import pyfits as pf
import pylab as plt
import RVTools as RVT

fileList = glob.glob('cam1/*.fits')
b=[]
wl = RVT.extract_HERMES_wavelength(fileList[0])
#build fibre filter
a = pf.open(fileList[0])
filt = np.logical_or((a['FIBRES'].data['TYPE']=='P'),(a['FIBRES'].data['TYPE']=='S'))
filt[175]=False

plt.plot(wl,np.sum(a[0].data[filt], axis=0))
plt.show()



# for fits in fileList[:]:
#     print 'Reading',fits
#     a = pf.getdata(fits)
#     if b==[]:
#         b =a
#     else:
#         b+=a



# ### SNR 3d plots

# In[ ]:

SNRs=np.load('npy/SNRs.npy')
Data=np.load('npy/Data.npy')

labels = Data[:,0]


# In[ ]:

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
X = np.arange(SNRs.shape[1])
Y = np.arange(SNRs.shape[0])
X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
Z = SNRs[:,:,0]
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0,  vmin=0, vmax=100, antialiased=True)
ax.set_xlabel('Epoch')
ax.set_ylabel('Star')
ax.set_zlabel('SNR')
# ax.set_zlim(-1.01, 1.01)
ax.set_yticks(np.arange(0,SNRs.shape[0],5))
# ax.set_yticklabels(labels)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()



# ### Comments array

# In[ ]:

def comment(star, epoch, cam, comment):
    comments = []
    try:
        comments = np.load('npy/comments.npy')
    except:
        pass
    
    if comments==[]:
        comments = np.zeros((1,),dtype=('i4,i4,i4,a10'))
        comments[:] = [(star, epoch, cam, comment)]
    else:
        x = np.zeros((1,),dtype=('i4,i4,i4,a10'))
        x[:] = [(star, epoch, cam, comment)]
        print x,comments
        comments = np.append(comments,x)
    
    np.save('npy/comments.npy',comments)
        


# In[ ]:

comment(0,0,0,'test')


# In[ ]:

c = np.load('npy/comments.npy') 
d = np.load('npy/data.npy')


# In[ ]:

d[1]


# In[ ]:

filename = 'obj/Field03.obj'
# filename = 'red_Giant01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)


# In[ ]:

from scipy import stats


# In[ ]:

thisCam = thisStar.exposures.cameras[0]


# In[ ]:

stats.nanmedian(thisCam.red_fluxes[13])


# In[ ]:

c


# In[ ]:

[np.asarray(a), np.asarray(a)]


# In[ ]:

x = np.zeros((2,),dtype=('i4,i4,i4,a10'))
x[:] = [(1,2,3,'Hello'),(2,3,4,"World")]


# In[ ]:

np.append(x,x)


# In[ ]:

np.vstack((x,(1,2,3,'Hello')))


# ### Check RVCorr

# In[ ]:

data=np.load('npy/data.npy')
RVs=np.load('npy/RVs.npy')
sigmas=np.load('npy/sigmas.npy')
baryVels=np.load('npy/baryVels.npy')
JDs=np.load('npy/JDs.npy')    
# RVCorr_PM=np.load('npy/RVCorr_PM.npy')
# # RVCorr_DM=np.load('npy/RVCorr_DM.npy')
# cRVs_PM=np.load('npy/cRVs_PM.npy')
# cRVs_DM=np.load('npy/cRVs_DM.npy')
# cRVs_PMDM=np.load('npy/cRVs_PMDM.npy')


# In[ ]:

idx = np.where(data[:,0]=='Giant01')[0][0]


# In[ ]:

starIdx = idx
cam = 0

RVs[RVs>5000]=np.nan
RVs[RVs<-5000]=np.nan


# In[ ]:

plt.plot(RVs[starIdx,:,cam])
# plt.plot(RVCorr_DM[starIdx,:,cam])
plt.plot(RVCorr_PM[starIdx,:,cam])
# plt.plot(RVs[starIdx,:,cam]-RVCorr_DM[starIdx,:,cam])
plt.plot(RVs[starIdx,:,cam]-RVCorr_PM[starIdx,:,cam])
plt.show()


# In[ ]:

reload(RVT)


# In[ ]:

allW_PM = RVT.create_allW(data, SNRs, starSet = [], RVCorrMethod = 'PM', refEpoch = 0) 
# RVCorr_PM = RVT.create_RVCorr_PM(RVs, allW_PM, RVClip = 2000, starSet = [])


# In[ ]:

plt.plot(allW_PM[:,0,starIdx])
plt.plot(SNRs[:,0,starIdx])
plt.plot(W)
plt.show()


# In[ ]:

data[starIdx]


# In[ ]:

p2y = RVT.pivot_to_y('/Users/Carlos/Documents/HERMES/reductions/6.2/rhoTuc_6.2/0_20aug/1/20aug10042tlm.fits') 
datay = p2y[data[:,2].astype(float).astype(int)]
deltay = datay-datay[starIdx]

thisSigma = 1./SNRs[:,0,0].copy()
thisSigma[np.isnan(thisSigma)]=1e+17  #sets NaNs into SNR=1e-17
W = RVT.calibrator_weights(deltay,thisSigma)


# In[ ]:

W


# In[ ]:

import psycopg2 as mdb
con = mdb.connect("dbname=hermes_master user=Carlos")
cur = con.cursor()
cur.execute("CREATE TABLE fields(id int)")


# In[ ]:

con.rollback()


# In[ ]:

con.commit()


# In[ ]:

con.close()


# In[ ]:

import psycopg2 as mdb
con=mdb.connect("host=/tmp/ dbname=hermes_master user=Carlos");


# In[ ]:

con=mdb.connect("dbname=hermes_master user=Carlos");


# In[ ]:


con=mdb.connect("host=/usr/local/var dbname=hermes_master user=Carlos");


# In[ ]:

cur = con.cursor()


# In[ ]:

cur.execute("SELECT spec_path,name from fields where ymd=140825 and ccd='ccd_1' and obstype='BIAS'")


# In[ ]:

objs=cur.fetchall()


# In[ ]:

from pyraf import iraf


# In[ ]:

iraf.noao(_doprint=0,Stdout="/dev/null")
iraf.imred(_doprint=0,Stdout="/dev/null")
iraf.ccdred(_doprint=0,Stdout="/dev/null")


# In[ ]:

iraf.ccdproc(images='tmp/flats/25aug10034.fits', ccdtype='', fixpix='no', oversca='no', trim='no', zerocor='yes', darkcor='no', flatcor='no', zero='tmp/masterbias',Stdout="/dev/null")


# In[ ]:

pwd


# In[ ]:

cd ~/Documents/workspace/GAP/IrafReduction/140825/ccd11/


# In[ ]:

import pyfits as pf


# In[ ]:

pf.open('tmp/masterbias.fits')


# In[ ]:

import cosmics


# In[ ]:

import numpy as np
import pylab as plt


# In[ ]:

a = np.arange(5)
b = np.array([1,5,3,2,5])
c = np.arange(0.5,4.5)


# In[ ]:

d = np.interp(c,a,b)


# In[ ]:

b


# In[ ]:

plt.plot(a,b,marker='+')
plt.scatter(c,d, marker = '+', s=200, c='r')
plt.show()


# In[ ]:

import os
import create_obj as cr_obj
import pickle
import glob
import pyfits as pf
import numpy as np
import sys
import toolbox
import importlib


# In[ ]:

booHD1581 = False
IRAFFiles = '/Users/Carlos/Documents/workspace/GAP/IrafReduction/results/'   #folder to IRAF reduced files
dataset = 'HD285507'


# In[ ]:

os.mkdir('cam1')
os.mkdir('cam2')
os.mkdir('cam3')
os.mkdir('cam4')
os.mkdir('obj')


# In[ ]:

thisDataset = importlib.import_module('data_sets.'+dataset)

    


# In[ ]:

months = np.array(['', 'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
d = np.array([s[4:] for s in thisDataset.date_list])
m = months[np.array([s[2:4] for s in thisDataset.date_list]).astype(int)]
filename_prfx = np.core.defchararray.add(d, m)


# In[ ]:

for i,folder in enumerate(thisDataset.date_list):
    for files in thisDataset.ix_array[i][2:]:
        for cam in range(1,5):
            strCopy = 'cp ' + IRAFFiles + folder + '/norm/' + filename_prfx[i] + str(cam) + "%04d" % (files,) + '.ms.fits ' 
            strCopy += 'cam'+ str(cam) + '/' + filename_prfx[i] + str(cam) + "%04d" % (files,) + '.fits ' 
            print strCopy
            try:
                os.system(strCopy)
            except:
                print 'no copy'


# In[ ]:

thisDataset.ix_array[1][2:]


# In[ ]:




# In[ ]:

filename_prfx


# In[ ]:

thisDataset.ix_array


# ### CC arcs

# In[ ]:

cd ~/Documents/HERMES/reductions/6.5/HD285507/


# In[ ]:

import glob
import os
import importlib
import numpy as np


# In[ ]:

#Copy are files
files = glob.glob('*')
thisDataset = importlib.import_module('data_sets.HD1581')

# for folderList in files:
#     try:
#     if int(folderList[:1]) in range(20):

#compose file prefixes from date_list
months = np.array(['', 'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
d = np.array([s[4:] for s in thisDataset.date_list])
m = months[np.array([s[2:4] for s in thisDataset.date_list]).astype(int)]
filename_prfx = np.core.defchararray.add(d, m)



for i,folder in enumerate(thisDataset.date_list):
    thisFile = "%04d" % thisDataset.ix_array[i][1]
    for cam in range(1,5):
        strCopy = 'cp ' + str(i) + '_' + filename_prfx[i] + '/' + str(cam) + '/' + filename_prfx[i] + str(cam) + thisFile + 'red.fits ' 
#         strCopy = str(i) + '_' + filename_prfx[i] + '/' + str(cam) + '/' + filename_prfx[i] + str(cam) + thisFile + 'red.fits ' 
#         strCopy += 'arc_cam'+ str(cam) + '/' + filename_prfx[i] + str(cam) + thisFile + 'red.fits ' 
        strCopy += '.' 
        print strCopy
        
        os.system(strCopy)

            
#     except:
#         print 'error'


# In[ ]:

import pyfits as pf
import pylab as plt
import numpy as np
import RVTools as RVT
from scipy import signal, optimize, constants
import os
import glob
reload(RVT)

os.chdir('/Users/Carlos/Documents/HERMES/reductions/6.5/HD1581/')

corrHWidth = 5
xDef = 1
fibre = 30

arcRVs = np.ones((400,5,4))*np.nan

for cam in range(4):
    for fibre in [175]: range(400):
        print fibre
        files = glob.glob('arc_cam'+str(cam+1)+'/*')
        for i,thisFile in enumerate(files):
            if i==0:
                fits = pf.open(thisFile)
                refWL = RVT.extract_HERMES_wavelength(thisFile)
                refData = fits[0].data
                fits.close()

        #         print refWL.shape
        #         print refData.shape
        #         plt.plot(refWL, refData[101])
        #         plt.show()


            fits = pf.open(thisFile)
            thisWL = RVT.extract_HERMES_wavelength(thisFile)
            thisData = fits[0].data
            fits.close()

            lambda1, flux1 = RVT.clean_flux(refWL, refData[fibre], flatten = False)
    #         plt.plot(lambda1,flux1)
            lambda2, flux2 = RVT.clean_flux(thisWL, thisData[fibre], flatten = False)
    #         plt.plot(lambda2,flux2)
    #         plt.show()

            try:
                CCCurve = signal.fftconvolve(flux1[-np.isnan(flux1)], flux2[-np.isnan(flux2)][::-1], mode='same')
                corrMax = np.where(CCCurve==max(CCCurve))[0][0]
                p_guess = [corrMax,corrHWidth]
                x_mask = np.arange(corrMax-corrHWidth, corrMax+corrHWidth+1)
                if max(x_mask)<len(CCCurve):
                    p = fit_gaussian(p_guess, CCCurve[x_mask], np.arange(len(CCCurve))[x_mask])[0]
                    if np.modf(CCCurve.shape[0]/2.0)[0]>1e-5:
                        pixelShift = (p[0]-(CCCurve.shape[0]-1)/2.) #odd number of elements
                    else:
                        pixelShift = (p[0]-(CCCurve.shape[0])/2.) #even number of elements


                    mid_px = refData.shape[1]/2
                    dWl = (refWL[mid_px+1]-refWL[mid_px]) / refWL[mid_px]/xDef
                    RV = dWl * pixelShift * constants.c 
    #                 print 'RV',fibre,i,RV
                    arcRVs[fibre,i,cam] = RV
            except:
                pass


# In[ ]:

cam =3
filename = 'HD1581arc_IR'

files = glob.glob('arc_cam'+str(cam+1)+'/*')
for i,thisFile in enumerate(files):
    fits = pf.open(thisFile)
    refWL = RVT.extract_HERMES_wavelength(thisFile)
    refData = fits[0].data
    fits.close()
    
    file_object = open(filename+'_e'+str(i)+'.txt', 'w')
    for wl,fl in zip(refWL,refData[175]):
        file_object.write(str(wl)+' '+str(fl)+'\n')
    file_object.close()


# In[ ]:

import pickle
filename = '../obj/HD1581.obj'
# filename = 'Brght01.obj'
# filename = 'red_Giant01.obj'
# filename = 'Giant01.obj'
# filename = 'Field01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)
thisCam =  thisStar.exposures.cameras[0]

thisStar.exposures.pivots


# In[ ]:

RVs = np.nanmean(arcRVs,axis=0)


# In[ ]:

X = JDs[np.array([0,1,4,7,12])]


# In[ ]:

JDs = np.load('../npy/JDs.npy')
plt.scatter(JDs,arcRVs[175])
plt.show()


# In[ ]:

np.save('npy/arcRVs',arcRVs)


# In[ ]:

print arcRVs.shape
arcRVs2 = np.ones((400,15,4))*np.nan
arcRVs2[:,0,:] = arcRVs[:,0,:]
arcRVs2[:,1,:] = arcRVs[:,1,:]
arcRVs2[:,2,:] = arcRVs[:,1,:]
arcRVs2[:,3,:] = arcRVs[:,1,:]
arcRVs2[:,4,:] = arcRVs[:,2,:]
arcRVs2[:,5,:] = arcRVs[:,2,:]
arcRVs2[:,6,:] = arcRVs[:,2,:]
arcRVs2[:,7,:] = arcRVs[:,3,:]
arcRVs2[:,8,:] = arcRVs[:,3,:]
arcRVs2[:,9,:] = arcRVs[:,3,:]
arcRVs2[:,10,:] = arcRVs[:,3,:]
arcRVs2[:,11,:] = arcRVs[:,3,:]
arcRVs2[:,12,:] = arcRVs[:,4,:]
arcRVs2[:,13,:] = arcRVs[:,4,:]
arcRVs2[:,14,:] = arcRVs[:,4,:]
arcRVs = arcRVs2
print arcRVs.shape


# In[ ]:

arcRVs.shape


# In[ ]:

cd '/Users/Carlos/Documents/HERMES/reductions/6.5/HD1581/'


# In[ ]:

# arcRVs = np.load('npy/arcRVs.npy')
JDs = np.load('npy/JDs.npy')



# In[ ]:

JDs


# In[ ]:

arcRVs2 = np.ones((400,16,4))*np.nan
arcRVs2[:,0,:] = arcRVs[:,0,:]
arcRVs2[:,1,:] = arcRVs[:,0,:]
arcRVs2[:,2,:] = arcRVs[:,0,:]
arcRVs2[:,3,:] = arcRVs[:,0,:]
arcRVs2[:,4,:] = arcRVs[:,1,:]
arcRVs2[:,5,:] = arcRVs[:,1,:]
arcRVs2[:,6,:] = arcRVs[:,1,:]
arcRVs2[:,7,:] = arcRVs[:,2,:]
arcRVs2[:,8,:] = arcRVs[:,2,:]
arcRVs2[:,9,:] = arcRVs[:,2,:]
arcRVs2[:,10,:] = arcRVs[:,3,:]
arcRVs2[:,11,:] = arcRVs[:,3,:]
arcRVs2[:,12,:] = arcRVs[:,3,:]
arcRVs2[:,13,:] = arcRVs[:,4,:]
arcRVs2[:,14,:] = arcRVs[:,4,:]
arcRVs2[:,15,:] = arcRVs[:,4,:]
arcRVs = arcRVs2
np.save('npy/arcRVs',arcRVs)


# In[ ]:

bary = np.load('npy/baryVels.npy')


# In[ ]:




# In[ ]:

bary.shape


# In[ ]:




# In[ ]:

bary


# In[ ]:

bary[:-1]-bary[1:]


# In[ ]:

pwd


# In[ ]:

a = np.array([4860,4865])
np.save('npy/cam1Filter.npy',a)
a = np.array([5751,5756])
np.save('npy/cam2Filter.npy',a)
a = np.array([6560,6565])
np.save('npy/cam3Filter.npy',a)
a = np.array([7710,7718])
np.save('npy/cam4Filter.npy',a)


# In[ ]:

1.400E7
648.36
818.28
570.57
142.85
150.76
150.88
80.39
84.36
99.99
105.94
124.30
122.21
122.19
121.61


# In[ ]:

hbetaRVs = (np.loadtxt('out3_hbeta.txt', delimiter = ' ', dtype='str')[:,32]).astype(float)*1000


# In[ ]:

arcRVs = np.loadtxt('arc_hbeta.txt', delimiter = ' ', dtype='str', usecols = [46]).astype(float)*1000


# In[ ]:

bary = np.load('npy/baryVels.npy')
JDs = np.load('npy/JDs.npy')


# In[ ]:

np.loadtxt('arc_hbeta.txt', delimiter = ' ', dtype='str', usecols = [46])


# In[ ]:

[0,1,4,7,12]


# In[ ]:

plt.scatter(JDs,hbetaRVs,color = 'k', s=100, marker='*', label = 'Stars')
plt.scatter(JDs[np.array([0,1,4,7,12])],arcRVs,  marker = '+' , label = 'ARC RV', color = 'm', s=500)
plt.show()


# In[ ]:

arcRVs[0] = 1.4e-7


# In[ ]:

bary


# ### sine wave

# In[ ]:

cd /Users/Carlos/Documents/HERMES/reductions/6.5/rhoTuc/


# In[ ]:

import numpy as np
import pylab as plt
from scipy.optimize import leastsq


# In[ ]:

RVs = np.load('npy/RVs.npy')
baryVels = np.load('npy/baryVels.npy')
MJDs = np.load('npy/MJDs.npy')
data = np.load('npy/data.npy')


# In[ ]:

np.where(data[:,0]=='Giant01')
print data[15]


# In[ ]:

starIdx = 15
thisBaryRVs = (RVs[starIdx,:,0] - baryVels)[-np.isnan(RVs[starIdx,:,0])]
thisMJDs = MJDs[-np.isnan(RVs[starIdx,:,0])]


print 'Calculating RV fit for', thisBaryRVs.shape[0], 'data points'



# In[ ]:

def optimise_sine(x):
    
#     print 'x',x
    result = x[0]*(np.sin(np.pi*2./x[2]*(MJDs+x[1]))-np.sin(np.pi*2./x[2]*(MJDs[0]+x[1]))) - thisBaryRVs
    
    
    return result 


# In[ ]:

# data = 3.0*np.sin(t+0.001) + 0.5 + np.random.randn(N) # create artificial data with noise

# guess_mean = np.mean(thisBaryRVs)
guess_std = 3*np.std(thisBaryRVs)/(2**0.5)
guess_std = 40000
guess_A = np.abs((np.max(thisBaryRVs)-np.min(thisBaryRVs))/2.)
guess_phase = 0.
guess_P = 5.

print 'Initial x',guess_A,guess_phase,guess_P

# we'll use this to plot our first estimate. This might already be good enough for you
data_first_guess = guess_A*np.sin(2*np.pi/guess_P*(MJDs+guess_phase))

# plt.scatter(MJDs, thisBaryRVs)
# plt.plot(MJDs, data_first_guess)
# plt.title('First Guess')
# plt.show()

# Define the function to optimize, in this case, we want to minimize the difference
# between the actual data and our "guessed" parameters
# optimize_func = lambda x: (x[0]*np.sin(MJDs+x[1])-x[0]*np.sin(MJDs+x[1])[0]) - data


ouput = leastsq(optimise_sine, [guess_A, guess_phase, guess_P], full_output = True, diag = [1,100,100])
print np.std(ouput[2]['fvec'])

# est_std, est_phase, est_P =ouput[0]

# # recreate the fitted curve using the optimized parameters
# data_fit = est_std*np.sin(2*np.pi/est_P*(MJDs+est_phase))

# plt.plot(MJDs,thisBaryRVs, '.')
# plt.plot(MJDs,data_fit-data_fit[0], label='after fitting')
# # plt.plot(MJDs,data_first_guess, label='first guess')
# plt.title(('RV:'+str(est_std)+' ph:'+str(est_phase)+' P:'+str(est_P)))
# plt.legend()
# plt.show()


# In[ ]:


x[0]*(np.sin(np.pi*2./p*MJDs+X[1])-np.sin(np.pi*2./p*MJDs[0]+X[1])) - thisBaryRVs


# In[ ]:

data_first_guess


# In[ ]:

a = np.arange(4*np.pi+1)
p=np.pi
b = np.sin(np.pi*2./p*a)

plt.plot(a,b)
plt.show()


# ### tree plot

# In[ ]:

cd HERMES/reductions/6.5/47Tuc_core/


# In[ ]:

import pylab as plt
import numpy as np

zoom = 1
sigmaClip = -1
RVClip = -1
booSave = False
booShow = True
booBaryPlot = False
booBaryCorrect = False
title = ''
colors = ['b','g','r','cyan']
labels = ['Blue','Green','Red','IR']

    
data=np.load('npy/data.npy')
RVs=np.load('npy/RVs.npy')
#     sigmas=np.load('npy/sigmas.npy')
baryVels=np.load('npy/baryVels.npy')
#     JDs=np.load('npy/JDs.npy')
baryVels3D = np.zeros(RVs.shape)
baryVels3D[:,:,0] = np.tile(baryVels,[RVs.shape[0],1])
baryVels3D[:,:,1] = np.tile(baryVels,[RVs.shape[0],1])
baryVels3D[:,:,2] = np.tile(baryVels,[RVs.shape[0],1])
baryVels3D[:,:,3] = np.tile(baryVels,[RVs.shape[0],1])
baryRVs = RVs - baryVels3D


order = np.argsort(np.ptp(baryRVs,1),0)
print 'order',order
print 'barys',np.ptp(baryRVs,1)
# print '1',baryRVs[:,:,0]
# print '2',baryRVs[:,:,0][order[:,0]]

for i,line in enumerate(baryRVs[:,:,0]):
    print i, line, np.ptp(line)
    
    
    
plt.boxplot((baryRVs[:,:,0][order[:,0]]).transpose())
plt.show()
    
# #     if RVClip>-1:Y[np.abs(Y)>RVClip] = np.nan
# #     if sigmaClip>-1:
# #         stdY= np.std(Y)
# #         medY = np.median(Y)
# #         Y[(Y>=medY-sigmaClip*stdY) & (Y<=medY+sigmaClip*stdY)] = np.nan
    
# #     YERR = sigmas

#     X = np.arange(data[:,0].shape[0])
# #     bcRVs = RVs-baryVels
    
# #     Y = RVs
# #     Y[Y==0.]=np.nan
# #     order = np.argsort(np.nanstd(RVs,axis=1),axis=0)    
#     order = np.argsort(np.ptp(baryVels3D,1),0)
    
#     if zoom > 1 :
#         zoomList = [np.arange(len(X))]
    
#     steps = np.zeros(zoom+1)
#     for i in range(zoom+1):
#         steps[i] = len(X)/zoom*i
        
#     for i in range(zoom):
#         print 'i.zoom',i,zoom
#         if i<(zoom-1):
#             zoomList.append([np.arange(steps[i],steps[i+1]).astype(int)])
#         else:
#             print 'last zoom'
#             zoomList.append([np.arange(steps[i],len(X)).astype(int)])
    
#     zoomList = np.array(zoomList)
    
#     print 'steps,zoomList',steps,zoomList
#     for cam in range(4)[:1]:
#         for i, thisXRange in enumerate(zoomList):
            
#             print 'thisXRange',thisXRange
#             thisOrder = order[:,cam][thisXRange]
#             print 'thisOrder',thisOrder
            
#             fig, ax = plt.subplots()
#             ax.set_xticklabels(data[:,0][thisOrder])
#             ax.set_xticks(X[range(len(thisOrder))])
#             plt.xticks(rotation=90)

# #             if booBaryPlot==True: plt.plot(X, baryVels, label = 'Barycentric Vel. ')


#             thisY = baryRVs[:,:,cam][thisOrder]
#             thisY[thisY==0.]=np.nan

# #             YERR[:,:,cam] = YERR[:,:,cam][order[:,cam]]

#             #median
#             plt.scatter(X[range(len(thisOrder))], stats.nanmedian(thisY , axis = 1), label = labels[cam], color = 'k')

#             #sigma
#             plt.scatter(X[range(len(thisOrder))], stats.nanmedian(thisY, axis = 1)+stats.nanstd(thisY, axis = 1), label = labels[cam], color = 'r')
#             plt.scatter(X[range(len(thisOrder))], stats.nanmedian(thisY, axis = 1)-stats.nanstd(thisY, axis = 1), label = labels[cam], color = 'r')


#             #min max
#             for star in range(thisY.shape[0]):
#                 x = np.nanmax(thisY[star,:])
#                 n = np.nanmin(thisY[star,:])
#                 plt.plot([star,star],[x,n], color = 'g', lw=2)
#                 print star, n, x, x-n

#             #zero
#             plt.plot(X[range(len(thisOrder))],np.zeros(len(thisOrder)), '--')

#             plt.grid(axis='x')
#             plt.xlabel('Stars')
#             plt.ylabel('RV [m/s]')
#         #     plt.legend(loc=0)

#             fig.tight_layout()
#             if booSave==True: 
#                 try:
#                     plotName = 'plots/Tree_'+str(i)+'_'+labels[cam]
#                     print 'Attempting to save', plotName
#                     plt.savefig(plotName)

#                 except:
#                     print 'FAILED'
#             if booShow==True: plt.show()
#             plt.close()        
        


# In[ ]:

a = 432.4323423534235345


# In[ ]:

"%.2f" % a


# ### log resample tests

# In[ ]:

cd ~/Documents/HERMES/reductions/6.5/m67_lr/obj/


# In[ ]:

import pickle
import pylab as plt
import RVTools as RVT
import numpy as np
from scipy import optimize
filename = 'red_Giant01.obj'
filename = 'M67-590_10_0_56643.6659144.obj'

# filename = 'Field01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

thisCam = thisStar.exposures.cameras[0]


# In[ ]:


for i,thisCam in enumerate(thisStar.exposures.cameras):
        print i, thisCam.wavelengths
        print i, thisCam.red_fluxes
        


# ### HERPY RVS test1

# In[8]:

cd ~/Documents/HERMES/reductions/myherpy/HD1581/


# In[9]:

import pickle
filename = 'obj/red_Giant01.obj'
filename2 = 'obj/red_ThXe.obj'
filehandler = open(filename, 'r')
filehandler2 = open(filename2, 'r')
thisStar = pickle.load(filehandler)
thisStar2 = pickle.load(filehandler2)

thisCam = thisStar.exposures.cameras[0]
thisCam2 = thisStar2.exposures.cameras[0]


# In[10]:

thisCam.RVs -  thisCam2.RVs+ thisStar.exposures.rel_baryVels


# In[ ]:

import pylab as plt

plt.plot(thisStar.exposures.MJDs, thisCam.RVs,'.', label = 'Star')
plt.plot(thisStar.exposures.MJDs, thisStar.exposures.rel_baryVels, label = 'bary_star')
plt.plot(thisStar2.exposures.MJDs, thisCam2.RVs,'.', label = 'ThXe')
plt.plot(thisStar2.exposures.MJDs, thisStar2.exposures.rel_baryVels, '.',label = 'bary_ThXe')
plt.legend(loc=0)
plt.show()


# In[11]:

a = np.loadtxt('ThXe_prepared.txt')
wl = a[:,0]
Th = a[:,1:]


# In[74]:

b = np.loadtxt('Spec_prepared.txt')
wl_sp = b[:,0]
fl_sp = b[:,1:]



# In[162]:

c = np.loadtxt('MJD_RVcorr.txt')
MJD = c[:,0]-2400000.50
rel_bary = c[:,1]
print thisStar.exposures.MJDs- MJD
good_bary = (rel_bary-rel_bary[0])*1000


# In[142]:

from scipy import stats, constants, optimize, signal
import RVTools as RVT
reload(RVT)
import numpy as np
import pylab as plt

RV1= []
def clean_flux(wavelength, flux, minWL=0, maxWL=0, xStep = 5*10**-6, medianRange = 5, flatten = False):
    
    #fix initial nans on edges
    nanMap = np.isnan(flux)
    leftEdgeIdx=0
    rightEdgeIdx=len(flux)
    
#     nanMapIdx = np.where(nanMap==True) <<<<<make the next lines faster by using this
    if np.sum(nanMap)>0:
        print 'Found NaNs in flux array'
        
    for i,booI in enumerate(nanMap):
        if booI==False:
            leftEdgeIdx = i
            break
            
    for j,rbooI in enumerate(nanMap[::-1]):
        if rbooI==False:
            rightEdgeIdx = len(nanMap)-j
            break        

    fluxMedian = stats.nanmedian(flux)
    if leftEdgeIdx>0:
        flux[:leftEdgeIdx] = np.linspace(fluxMedian, flux[leftEdgeIdx+1],leftEdgeIdx)
    if rightEdgeIdx<len(flux):
        flux[rightEdgeIdx:] = np.linspace(flux[rightEdgeIdx-1], fluxMedian, len(flux)-rightEdgeIdx)

        
        
    #median outliers
    if medianRange>0:
        fluxMed = signal.medfilt(flux,medianRange)
#         fluxDiff = abs(flux-fluxMed)
        fluxDiff = flux-fluxMed
        fluxDiffStd = np.std(fluxDiff)
        mask = fluxDiff> 3 * fluxDiffStd
        flux[mask] = fluxMed[mask]


    if ((wavelength[-np.isnan(flux)].shape[0]>0) &  (flux[-np.isnan(flux)].shape[0]>0)):
        
#         if flatten==True:#flatten curve by fitting a 3rd order poly
#             fFlux = optimize.curve_fit(cubic, wavelength[-np.isnan(flux)], flux[-np.isnan(flux)], p0 = [1,1,1,1])
#             fittedCurve = cubic(wavelength, fFlux[0][0], fFlux[0][1], fFlux[0][2], fFlux[0][3])
#             flux = flux/fittedCurve-1
#         else:
#             flux = flux/fluxMedian-1
            
        #apply tukey
        flux = flux * signal.tukey(len(flux), 0.1)

        #resample
#         print wavelength, flux, minWL, maxWL, xStep
        wavelength,flux = RVT.resample_sp(wavelength, flux, minWL, maxWL, xStep)
        
    else: #if not enough data return NaNs
        wavelength = np.ones(4096)*np.nan
        flux = np.ones(4096)*np.nan
        
    return wavelength, flux


#Create cross correlation curves wrt epoch 0

minWL, maxWL = np.min(wl_sp), np.max(wl_sp)

lambda1, flux1 = clean_flux(wl_sp, fl_sp[:,0], minWL, maxWL)

for i in range(15):

    lambda2, flux2 = clean_flux(wl_sp, fl_sp[:,i], minWL, maxWL)

    #Duncan's approach to CC. 
    CCCurve = np.correlate(flux1, flux2, mode='full')
    
    width = 15
    y = CCCurve[int(CCCurve.shape[0]/2.)-width:int(CCCurve.shape[0]/2.)+1+width].copy()
    y /=np.max(y)
    x = np.arange(-width,width+1)
    p,_ = fit_flexi_gaussian([1,3.,2.,1.,0],y,x )
#     x, mu, sig, power, a, d 
    plt.plot(x,y)
    x_dense = np.linspace(min(x),max(x))
    plt.plot(x_dense,flexi_gaussian(x_dense,p[0],p[1],p[2],p[3],p[4]), label='gaussian')
    plt.legend(loc=0)
    plt.show()
    shift = p[0]
    print p[0], 
#                 thisQ, thisdRV = QdRV(thisCam.wavelengths[epoch], thisCam.red_fluxes[epoch])

    px = 1000
    RV = (np.exp(lambda1[px+1]-lambda1[px]) -1) * constants.c * shift
    print 'RV',RV                
    RV1.append(RV)
    
# -2.93349067446e-07 RV -0.000439720289189
# -0.038707960129 RV -58.0219176085
# -0.178451980065 RV -267.493457415
# 0.315953211917 RV 473.603134055
# -0.65680488406 RV -984.528214371


# In[172]:

a = np.array(RV1)-np.array(RV2)[np.array([0,1,1,1,1,1,1,3,3,3,3,3,4,4,4])]-good_bary

print a.astype(int)
print good_bary


# In[131]:

RV2


# In[70]:

from scipy import stats, constants, optimize, interpolate
import RVTools as RVT
reload(RVT)
import numpy as np
import pylab as plt

RV2 =[]
def clean_arc(wavelength, flux, minWL=0, maxWL=0, xStep = 5 *10**-6, medianRange = 0, flatten = True):
    wavelength,flux = RVT.resample_sp(wavelength, flux, minWL, maxWL, xStep)
        
        
    return wavelength, flux



minWL, maxWL = np.min(wl), np.max(wl)

lambda1, flux1 = clean_arc(wl, Th[:,0], minWL, maxWL)

for i in range(5):

    lambda2, flux2 = clean_arc(wl, Th[:,i], minWL, maxWL)

    #Duncan's approach to CC. 
    CCCurve = np.correlate(flux1, flux2, mode='full')
    
    width = 15
    y = CCCurve[int(CCCurve.shape[0]/2.)-width:int(CCCurve.shape[0]/2.)+1+width].copy()
    y /=np.max(y)
    x = np.arange(-width,width+1)
    p,_ = fit_flexi_gaussian([1,3.,2.,1.,0],y,x )
#     x, mu, sig, power, a, d 
#     plt.plot(x,y)
#     x_dense = np.linspace(min(x),max(x))
#     plt.plot(x_dense,flexi_gaussian(x_dense,p[0],p[1],p[2],p[3],p[4]), label='gaussian')
#     plt.legend(loc=0)
#     plt.show()
    shift = p[0]
    print p[0], 
#                 thisQ, thisdRV = QdRV(thisCam.wavelengths[epoch], thisCam.red_fluxes[epoch])

    px = 1000
    RV = (np.exp(lambda1[px+1]-lambda1[px]) -1) * constants.c * shift
    print 'RV',RV                
    RV2.append(RV)


# In[42]:

print wl, 
print Th[:,0]
lnWavelength = np.log(wl)
print lnWavelength
fFlux = interpolate.splrep(lnWavelength, Th[:,0]) 
wavelength = np.arange(np.log(minWL), np.log(maxWL),5*10**-6)
print wavelength
flux = interpolate.splev(wavelength, fFlux)
plt.plot(np.log(wl),Th[:,0])
plt.plot(wavelength,flux)
plt.show()


# In[ ]:

duncan = [0, -58, -268, 474, -985]


# In[ ]:

print thisCam2.RVs


# In[ ]:

reload(RVT)
# for i in range(10):
i=0
wavelength = thisCam.wavelengths[i].copy()
flux = thisCam.red_fluxes[i].copy()
# wavelength2, flux2 = RVT.resample_sp(wavelength, flux)
wavelength3, flux3 = RVT.clean_flux(wavelength, flux, flatten = False, medianRange=9)

# wavelength3, flux3 = RVT.resample_sp(wavelength, flux, 10)
# print wavelength.shape, wavelength2.shape

# plt.plot(wavelength,flux)
# plt.plot(np.exp(wavelength2),flux2)
plt.plot(np.exp(wavelength3),flux3)
# plt.plot(flux)
# plt.plot(flux2)`
plt.show()



# In[ ]:

wavelength = thisCam.wavelengths[0].copy()
flux = thisCam.red_fluxes[0].copy()
wavelength1, flux1 = RVT.clean_flux(wavelength, flux, flatten = False, medianRange=9)
for i in range(15):
#     i=5
    wavelength = thisCam.wavelengths[i].copy()
    flux = thisCam.red_fluxes[i].copy()
    wavelength2, flux2 = RVT.clean_flux(wavelength, flux, flatten = False, medianRange=9)

    CCCurve = np.correlate(flux1, flux2, mode='full')

    y = CCCurve[int(CCCurve.shape[0]/2.)+1-5:int(CCCurve.shape[0]/2.)+1+4].copy()
    y /=np.max(y)
    x = np.arange(-4,5)
    p,_ = fit_gaussian([1,3.], y, x)
    shift = p[0]
    print p,
    plt.plot(x,y)
    plt.plot(gaussian(x,p[0], p[1]))
    plt.show()
    #redo this    
    px = 1000
    RV = (np.exp(wavelength2[px+1])-np.exp(wavelength2[px]))/np.exp(wavelength2[px])* c*shift
    print 'RV',RV                


# In[ ]:




# In[ ]:

#Fit gaussian in CCCurves
def gaussian(x, mu, sig, ):
    return np.exp(-np.power(x - mu, 2.) / 2 / np.power(sig, 2.))

def fit_gaussian(p, flux, x_range):
    a = optimize.leastsq(diff_gausian, p, args= [flux, x_range])
    return a

def diff_gausian(p, args):
    
    flux = args[0]
    x_range = args[1]
    diff = gaussian(x_range, p[0],p[1]) - flux/np.max(flux)
    return diff

def get_wavelength(wavelengths, pixel):
    intPx = int(pixel)
    fracPx = pixel - int(pixel)

    return (wavelengths[intPx+1] - wavelengths[intPx])*fracPx + wavelengths[intPx]

def extract_HERMES_wavelength(fileName):

    a = pf.open(fileName)

    CRVAL1 = a[0].header['CRVAL1'] # / Co-ordinate value of axis 1                    
    CDELT1 = a[0].header['CDELT1'] #  / Co-ordinate increment along axis 1             
    CRPIX1 = a[0].header['CRPIX1'] #  / Reference pixel along axis 1                   

    #Creates an array of offset wavelength from the referece px/wavelength
    Lambda = CRVAL1 - (CRPIX1 - (np.arange(int(CRPIX1)*2)))* CDELT1

    return Lambda


# ### pyHermes

# In[ ]:

cd /Users/Carlos/Documents/HERMES/reductions/


# In[ ]:

import pyfits as pf
import pylab as plt
import numpy as np


# In[ ]:

a = pf.open('pyhermes/initial_RV_bulk/140820HD285507_p1/1408200037_FIB228_1.fits')
b = pf.open('pyhermes/initial_RV_bulk/140820/ccd_1/20aug10039comb.fits')
file2_1 = pf.open('6.5/HD285507/cam1/20aug10039red.fits')


# In[ ]:

plt.plot(a[0].data/np.nanmax(a[0].data))
plt.plot(a[1].data/np.nanmax(a[1].data))
plt.show()


# In[ ]:

# plt.plot(a[0].data)
plt.plot(a[1].data)
# plt.plot(b[0].data[227])
plt.show()


# In[ ]:

import RVTools as RVT
reload(RVT)
RVT.extract_HERMES_wavelength(a.filename())
# a.filename()


# In[ ]:


thisFile = pf.open(a.filename())

CRVAL1 = thisFile[0].header['CRVAL1'] # / Co-ordinate value of axis 1                    
CDELT1 = thisFile[0].header['CDELT1'] #  / Co-ordinate increment along axis 1             
CRPIX1 = thisFile[0].header['CRPIX1'] #  / Reference pixel along axis 1                   
NAXIS1 = thisFile[0].header['NAXIS1'] #  / length of the array     
print CRVAL1,CDELT1,CRPIX1, NAXIS1

#Creates an array of offset wavelength from the referece px/wavelength
Lambda = (np.arange(int(NAXIS1)))* CDELT1 + CRVAL1

print Lambda
print thisFile[0].header['CDELT1']*(np.arange(1,thisFile[0].header['NAXIS1']+1,1)-thisFile[0].header['CRPIX1'])+thisFile[0].header['CRVAL1']
thisFile = pf.open(file2_1.filename())

CRVAL1 = thisFile[0].header['CRVAL1'] # / Co-ordinate value of axis 1                    
CDELT1 = thisFile[0].header['CDELT1'] #  / Co-ordinate increment along axis 1             
CRPIX1 = thisFile[0].header['CRPIX1'] #  / Reference pixel along axis 1       
print CRVAL1,CDELT1,CRPIX1


#Creates an array of offset wavelength from the referece px/wavelength
Lambda = CRVAL1 - (CRPIX1 - (np.arange(int(CRPIX1)*2)) -1)* CDELT1

print Lambda
print thisFile[0].header['CDELT1']*(np.arange(1,4097,1)-thisFile[0].header['CRPIX1'])+thisFile[0].header['CRVAL1']


# In[ ]:




# In[ ]:

from specutils.io import read_fits
myspec = read_fits.read_fits_spectrum1d(a.filename())


# In[ ]:

# plt.plot(myspec)
# plt.show()

myspec.dispersion()


# In[ ]:

from specutils.wcs import specwcs
from astropy.io import fits
from astropy import units as u
from specutils import Spectrum1D

header = fits.getheader(b.filename())
dispersion_start = header['CRVAL1'] - (header['CRPIX1'] - 1) * header['CDELT1']
linear_wcs = specwcs.Spectrum1DPolynomialWCS(degree=1, c0=dispersion_start, c1=header['CDELT1'], unit=u.Unit('Angstrom'))
flux = fits.getdata(a.filename())
myspec = Spectrum1D(flux=flux, wcs=linear_wcs)


# In[ ]:

linear_wcs.evaluate(range(10))


# In[ ]:

a[0].header.items()


# In[ ]:

b[1].header.items()


# In[ ]:

pwd


# In[ ]:

ls


# In[ ]:

cd ..


# In[ ]:

cd obj/


# In[ ]:

ls


# In[ ]:

import pickle
filename = 'HD1581.obj'
# filename = 'Field01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)


# In[ ]:

thisStar.exposures.JDs*1000


# In[ ]:




# In[1]:

2+2


# In[ ]:



