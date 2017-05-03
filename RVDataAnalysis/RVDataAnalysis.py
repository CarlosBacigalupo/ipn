
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import pylab as plt
import pyfits as pf
import RVTools as RVT

import astropy.units as u
from astropy.coordinates import SkyCoord


# In[2]:

def f1(x):
    return '%5.2f' % x


# ### 47tuc

# In[ ]:

cd ~/Documents/HERMES/reductions/47Tuc_core_6.2/npy


# In[ ]:

cd ~/Documents/HERMES/reductions/47Tuc_core_6.2/npy_150509/


# In[ ]:

cd ~/Documents/HERMES/reductions/47Tuc_core_1arc_6.2/npy/


# In[ ]:

cd ~/Documents/HERMES/reductions/47Tuc_core_1arc_6.2/npy_150509/


# In[ ]:

cd ~/Documents/HERMES/reductions/6.5/47Tuc_core/npy/


# ### HD1581

# In[ ]:

cd ~/Documents/HERMES/reductions/HD1581_6.2/npy/


# In[ ]:

cd ~/Documents/HERMES/data/140820/data/ccd_1/


# ### HD285507

# In[ ]:

cd ~/Documents/HERMES/reductions/HD285507_6.2/npy_150509/


# In[ ]:

cd ~/Documents/HERMES/reductions/HD285507_6.2/npy/


# In[ ]:

cd ~/Documents/HERMES/reductions/6.5/HD285507_1arc/npy/


# In[ ]:

cd ~/Documents/HERMES/reductions/HD285507_1arc_6.2/npy/


# In[ ]:

cd ~/Documents/HERMES/reductions/6.5/HD285507/npy/


# In[ ]:

cd ~/Documents/HERMES/data/140820/data/ccd_1/


# ### m67

# In[ ]:

cd ~/Documents/HERMES/reductions/m67_6.2/npy


# In[ ]:

cd ~/Documents/HERMES/reductions/m67_6.2/npy_raw/


# In[ ]:

cd ~/Documents/HERMES/reductions/m67_1arc_6.2/


# In[5]:

cd ~/Documents/HERMES/reductions/6.5/m67_lr/npy2/


# In[ ]:

cd ~/Documents/HERMES/data/140820/data/ccd_1/


# ### NGC2477

# In[ ]:

cd ~/Documents/HERMES/reductions/NGC2477_6.2/npy


# In[ ]:

cd ~/Documents/HERMES/reductions/NGC2477_6.2/npy_150509/


# In[ ]:

cd ~/Documents/HERMES/reductions/NGC2477_1arc_6.2/


# In[ ]:

cd ~/Documents/HERMES/reductions/6.5/NGC2477/


# ### rhoTuc

# In[ ]:

cd ~/Documents/HERMES/reductions/rhoTuc_6.2/npy


# In[ ]:

cd ~/Documents/HERMES/reductions/6.2/rhoTuc_1arc_6.2/npy


# ### Sine Fit Results

# In[ ]:

fittedData = np.load('fittedData.npy')
fittedResults = np.load('fittedResults.npy')
fittedRangeResults = np.load('fittedRangeResults.npy')
fittedAvgResults = np.load('fittedAvgResults.npy')
fittedAvgRangeResults = np.load('fittedAvgRangeResults.npy')

# RVs=np.load('RVs.npy')
# sigmas=np.load('sigmas.npy')
# baryVels=np.load('baryVels.npy')
# JDs=np.load('JDs.npy')

# thisSlice = RVs
# totalStars = RVs.shape[0]
# goodStars = np.sum(((thisSlice<5000) & (thisSlice!=0)), axis=0).astype(float)
# size = np.array(thisSlice.shape)
# total = np.reshape(np.repeat(size[0], size[1]*size[2]),(size[1],size[2]))
# labels = ['Blue','Green','Red','IR']

labels = ['Name',
          'Ratio Cam1','Ratio Cam2','Ratio Cam3','Ratio Cam4',
          'Range Cam1','Range Cam2','Range Cam3','Range Cam4',
          'Avg Ratio Cam1','Avg Ratio Cam2','Avg Ratio Cam3','Avg Ratio Cam4',
          'Avg Range Cam1','Avg Range Cam2','Avg Range Cam3','Avg Range Cam4' ]
# print fittedData.astype(str).shape, fittedResults.astype(str).shape
data1 =  fittedData
df1 = pd.DataFrame(data1)

data2 =  np.hstack((fittedResults, fittedRangeResults, fittedAvgResults, fittedAvgRangeResults))
df2 = pd.DataFrame(data2)


df = pd.concat([df1, df2], axis=1)
df.columns = labels

# print df.to_latex()
print df.loc[0:30][[0,1,5,9,13]].to_latex() #cam1
print '\\'+'newpage'
print df.loc[0:30][[0,2,6,10,14]].to_latex() #cam1
print '\\'+'newpage'
print df.loc[0:30][[0,3,7,11,15]].to_latex() #cam1
print '\\'+'newpage'
print df.loc[0:30][[0,4,8,12,16]].to_latex() #cam1


# ### % of stars within a given RV

# In[ ]:

data=np.load('data.npy')
RVs=np.load('RVs.npy')
sigmas=np.load('sigmas.npy')
baryVels=np.load('baryVels.npy')
JDs=np.load('JDs.npy')
thisSlice = RVs
totalStars = RVs.shape[0]
goodStars = np.sum(((thisSlice<5000) & (thisSlice!=0)), axis=0).astype(float)
size = np.array(thisSlice.shape)
total = np.reshape(np.repeat(size[0], size[1]*size[2]),(size[1],size[2]))
labels = ['Blue','Green','Red','IR']
a = pd.DataFrame(goodStars/total*100)
a.columns = labels

print a.to_latex(formatters=[f1, f1, f1, f1])


# In[ ]:

a = os.system('ls ')
print 


# ### Shows all observatios for a list of 'good_targets'

# In[ ]:

import pyfits as pf
import numpy as np
import pylab as plt
import os
import glob
import pickle
import RVTools as RVT
import toolbox

os.chdir('/Users/Carlos/Documents/HERMES/data')
# raw = []
print 'Obs & Filename & Field Name & Plate & MJD & Relative Day (days) & Exp Time (s) \\\\'
# good_targets = np.array(['rhoTuc #176 (Pivot 175)', 'HD1581 #176 (Pivot 175)','HD285507 #227 (Pivot 223)' ])
# good_targets = np.array(['rhoTuc #176 (Pivot 175)'])
# good_targets = np.array(['HD285507 #227 (Pivot 223)' ])
good_targets = np.array(['M67 12V14','M67 Bright Stars' ])
# good_targets = np.array(['HD1581 #176 (Pivot 175)'])
# good_targets = np.array(['47Tuc center'])

# good_pivots = [175,175,224]
t0 = 0  
order = 0 
dirs = glob.glob('*')
for j in dirs:
#     print os.path.abspath(j +'/data/ccd_1/')
#     os.chdir(j+'/data/ccd_1/')
#     print j +'/data/ccd_1/'
    for i in glob.glob(j +'/data/ccd_1/*.fits'):
        a = pf.open(i)
        if a[0].header['NDFCLASS'] == 'MFOBJECT':
            if a[0].header['OBJECT'] in good_targets:
                pivot_idx = np.where(a[0].header['OBJECT']== good_targets)[0][0]
                e = float(a[0].header['EXPOSED'])/2/24/60/60 # EXPOSED/2 in days
                inDate = np.hstack((a[0].header['UTDATE'].split(':'),a[0].header['UTSTART'].split(':'))).astype(int)
                UTMJD = toolbox.gd2jd(inDate, TZ=0)-2400000.5 + e
                if t0==0:t0=UTMJD
                tDiff=UTMJD-t0
#                 print str(order)+' & '+ i+' & HD1581 & '+a[0].header['SOURCE']+' & '+str(UTMJD)+' & '+str(tDiff)+' & '+str(a[0].header['EXPOSED']) +' \\\\'
#                 print str(order)+' & '+ i.split('/')[-1]+' & HD285507 & '+a[0].header['SOURCE']+' & '+str(UTMJD)+' & '+str(tDiff)+' & '+str(a[0].header['EXPOSED']) +' \\\\'
                print str(order)+' & '+ i.split('/')[-1]+' & '+ 'rhoTuc' +' & '+a[0].header['SOURCE']+' & '+str(UTMJD)+' & '+str(tDiff)+' & '+str(a[0].header['EXPOSED']) +' \\\\'
#                 print str(order)+' & '+ i.split('/')[-1]+' & '+a[0].header['OBJECT']+' & '+a[0].header['SOURCE']+' & '+str(UTMJD)+' & '+str(tDiff)+' & '+str(a[0].header['EXPOSED']) +' \\\\'
                order +=1
#                 raw.append((i, a[0].header['OBJECT'],a[0].header['SOURCE'],a[0].header['UTMJD'],a[0].header['EXPOSED']))            
        a.close
# raw = np.array(raw)


# ### Shows all observations

# In[6]:

import numpy as np
import pandas as pd


# In[7]:

cd ~/Documents/ipn/


# In[11]:

# outFileNames = np.load('npy/outFileNames.npy')
# outObjects = np.load('npy/outObjects.npy')
# outPlates = np.load('npy/outPlates.npy')
# outMJD_Exp = np.load('npy/outMJD_Exp.npy')

# outFileNames = np.delete(outFileNames,86)
# outObjects = np.delete(outObjects,86)
# outPlates = np.delete(outPlates,86)
# outMJD_Exp = np.delete(outMJD_Exp,86, axis = 0)


# np.save('npy/outFileNames.npy', outFileNames)
# np.save('npy/outObjects.npy', outObjects)
# np.save('npy/outPlates.npy', outPlates)
# np.save('npy/outMJD_Exp.npy', outMJD_Exp)


# In[8]:


outFileNames = np.load('outFileNames.npy')
outObjects = np.load('outObjects.npy')
outPlates = np.load('outPlates.npy')
outMJD_Exp = np.load('outMJD_Exp.npy')

ordIdx = np.argsort(outMJD_Exp[:,0])


labels = ['Obs', 'Filename','Field Name','Plate','MJD','Relative Day (days)','Exp Time (s)']
data = np.vstack((range(len(ordIdx)), outFileNames[ordIdx], outObjects[ordIdx], outPlates[ordIdx], outMJD_Exp[:,0][ordIdx], outMJD_Exp[:,0][ordIdx] - outMJD_Exp[:,0][ordIdx][0], outMJD_Exp[:,1][ordIdx])).transpose()

a = pd.DataFrame(data)
a.columns = labels

print a.to_latex(index=False)


# ### Stars Names and magnitudes

# In[ ]:

data=np.load('data.npy')
# RVs=np.load('RVs.npy')
# sigmas=np.load('sigmas.npy')
# baryVels=np.load('baryVels.npy')
# JDs=np.load('JDs.npy')
# thisSlice = RVs
# totalStars = RVs.shape[0]
# goodStars = np.sum(((thisSlice<5000) & (thisSlice!=0)), axis=0).astype(float)
# size = np.array(thisSlice.shape)
# total = np.reshape(np.repeat(size[0], size[1]*size[2]),(size[1],size[2]))
# labels = ['Blue','Green','Red','IR']
labels = ['Name','Vmag']
a = pd.DataFrame(data[:,:2])
a.columns = labels

print a.to_latex()


# ### W tests

# In[ ]:

#real data

method = 'PM'

data=np.load('npy/data.npy')

SNRs=np.load('npy/SNRs.npy')

if method=='PM':
    allW = np.load('npy/allW_PM.npy')
elif method=='DM':
    allW = np.load('npy/allW_DM.npy')
    
idx = np.where(data[:,0]=='Giant01')[0][0]

for cam in range(4):
    W = allW[:,cam,idx]

    thisSNRs = SNRs[:,0,cam]

#     plt.plot(deltay/np.max(np.abs(deltay)), label = 'deltay')
    plt.plot(thisSNRs, label= 'SNR', c='g')
    plt.plot(W*np.nanmax(thisSNRs), label = 'W', c='r')
    plt.plot([idx,idx],[0,np.nanmax(thisSNRs)], label = 'Ref. Star', c='cyan')
    plt.legend(loc=0)

    title = method + ' - cam '+str(cam)+ ' ,' + str(data[idx])
    
    plt.title(title)
    plt.grid(True)
    plt.savefig((method+'_'+str(cam)));plt.close()
#     plt.show()



# In[ ]:

#SIMULATION
import RVTools as RVT
reload(RVT)

deltays = np.zeros((3,50))
SNRss = np.zeros((3,50))
idxs = [1,20,40]

yrange = np.linspace(0, 4000)
# deltays[0] = 
# deltays[1] = np.linspace(-1999, 2000)
# deltays[1,25]=0
# deltays[2] = np.linspace(-3999, 0)
SNRss[0] = np.ones(50)*10
SNRss[1] = np.linspace(1, 10)
SNRss[2] = np.linspace(10, 1)

method = 'PM'

for i in range(3):
    for j in range(3):
        idx = idxs[j]
        deltay = yrange - yrange[idx]
        SNRs = SNRss[i]
        if method=='PM':
#             W = RVT.calibrator_weights2(deltay,SNRs)
            W = RVT.calibrator_weights(deltay,1/SNRs)
        elif method=='DM':
            W = RVT.calibrator_weights2(deltay,1/SNRs)

        plt.plot(deltay/np.max(np.abs(deltay)), label = 'deltay')
        plt.plot(SNRs, label= 'SNR')
        plt.plot(W*10, label = 'W')
        plt.plot([idx,idx],[0,np.nanmax(SNRs)], label = 'Ref. Star')

#         plt.plot(W2*100, label = 'W New')
        plt.legend(loc=0)
    
        title = method +' - Ref Star index = '+str(idx)+' - SNR '+str(np.min(SNRs))+','+str(np.max(SNRs))

        plt.title(title)
        plt.grid(True)
        figTitle = method+'_'+str(i)+'_'+str(j)
        plt.savefig(figTitle); plt.close()
#         plt.show()


# In[ ]:

SNRs=np.load('SNRs.npy')
import toolbox as tb


# In[ ]:

for i,a in enumerate(zip(data[:,],SNRs[:,0,0])):
    print i,a[0],a[1], tb.dec2sex(float(a[0][3])/15), tb.dec2sex(float(a[0][4]))


# ### All stars summary

# In[ ]:

#('NAME', 'S80'), ('RA', '>f8'), ('DEC', '>f8'), ('X', '>i4'), ('Y', '>i4'), ('XERR', '>i2'), ('YERR', '>i2'), ('THETA', '>f8'), ('TYPE', 'S1'), ('PIVOT', '>i2'), ('MAGNITUDE', '>f8'), ('PID', '>i4'), ('COMMENT', 'S80'), ('RETRACTOR', 'S10'), ('WLEN', '>f8'), ('PMRA', '>f8'), ('PMDEC', '>f8')


# In[ ]:

# a = pf.open('20aug10039.fits') #HD285507
a = pf.open('20aug10053.fits') #HD1581


# In[ ]:

good_targets_filter = (a[1].data['TYPE']=='P')
good_targets_idx = np.arange(400)[good_targets_filter]


# In[ ]:

outArray = []
for i in good_targets_idx:
    gc = SkyCoord(ra=a[1].data['RA'][i]*u.si.radian, dec=a[1].data['DEC'][i]*u.si.radian)
#     print a[1].data['RA'][i]*57/15
    name = a[1].data['NAME'][i]
    mag = a[1].data['MAGNITUDE'][i]
    RAhms = str(int(gc.ra.hms.h))+'h'+str(int(gc.ra.hms.m))+'m'+str(gc.ra.hms.s)+'s'
    RAdms = str(gc.ra)
    RAdeg = str(gc.ra.deg)
    DECdms = str(gc.dec)
    DECdeg = str(gc.dec.deg)
    pivot = a[1].data['PIVOT'][i]
    outArray.append((i, name, mag, RAhms, RAdms, RAdeg, DECdms, DECdeg, pivot))
labels = ['Index','Name','Kmag','RA(hms)','RA(dms)','RA(deg)','DEC(dms)','DEC(deg)','Pivot']
df = pd.DataFrame(outArray)
df.columns = labels

outLatex = df.to_latex()
print df.to_latex()


# In[ ]:

text_file = open("starsLatex.txt", "w")
text_file.write(outLatex)
text_file.close()


# ### All stars summary - from obj

# In[7]:

import glob
import pickle
import numpy as np
import pandas as pd


# In[4]:

cd ~/Documents/HERMES/reductions/6.5/m67_lr/obj_single/


# In[5]:

filehandler = open('M67-375.obj', 'r')
thisStar = pickle.load(filehandler)
thisCamera= thisStar.exposures.cameras[0]
thisCamera.fileNames
thisStar.exposures.plates[10][-1]
# print np.vstack((thisStar.exposures.pivots,np.nansum(thisCamera.red_fluxes,axis=1))).transpose()
# print thisStar.exposures.p


# In[8]:


outFileNames = np.load('/Users/Carlos/Documents/ipn/npy/outFileNames.npy')
# outObjects = np.load('npy/outObjects.npy')
# outPlates = np.load('npy/outPlates.npy')
outMJD_Exp = np.load('/Users/Carlos/Documents/ipn/npy/outMJD_Exp.npy')

outFileNames = np.delete(outFileNames,86)
# outObjects = np.delete(outObjects,86)
# outPlates = np.delete(outPlates,86)
outMJD_Exp = np.delete(outMJD_Exp,86, axis = 0)

ordIdx = np.argsort(outMJD_Exp[:,0])

outFileNames = outFileNames[ordIdx]
outMJD_Exp = outMJD_Exp[ordIdx]



# In[9]:

obsMatrix = np.ones((292, 99, 2, 4))*np.nan #stars, observations, [pivot, plate], cam

objList = glob.glob('*.obj')

outArray = []
i = 0
for thisObj in objList:
    if 'red' not in thisObj:
        filehandler = open(thisObj, 'r')
        thisStar = pickle.load(filehandler)
#         print thisStar.name
        
        gc = SkyCoord(ra=thisStar.RA*u.si.degree, dec=thisStar.Dec*u.si.degree)
        name = thisStar.name
        mag = thisStar.Vmag
        RAhms = str(int(gc.ra.hms.h))+'h'+str(int(gc.ra.hms.m))+'m'+str(f1(gc.ra.hms.s))+'s'
        RAdms = str(gc.ra)
        RAdeg = str(gc.ra.deg)
        DECdms = str(gc.dec)
        DECdeg = str(gc.dec.deg)
#         pivot = a[1].data['PIVOT'][i]
        i += 1
        ID = i
        outArray.append((ID, name, mag, RAhms, RAdms, RAdeg, DECdms, DECdeg))
        
        for cam in range(4):
#             print 'Camera',cam,
            thisCam = thisStar.exposures.cameras[cam]

            for exp in range(len(thisCam.fileNames)):
                if np.nansum(thisCam.red_fluxes[exp])!=0:
                    thisFileName = thisStar.exposures.cameras[0].fileNames[exp][:10]+'.fits'
#                     print thisFileName
                    fileIdx = np.where(outFileNames==thisFileName)[0]
                    if len(fileIdx)>0:
                        fileIdx = fileIdx[0]
                        obsMatrix[ID-1, fileIdx, 0, cam] = thisStar.exposures.pivots[exp]
                        obsMatrix[ID-1, fileIdx, 1, cam] = thisStar.exposures.plates[exp][-1]
#                         print 'Found', ID, fileIdx, cam
            
            
            
            
        filehandler.close()
        thisStar = None

outArray = np.array(outArray)
np.save('outArray.npy',outArray)

labels = ['ID', 'Name','Kmag','RA(hms)','RA(dms)','RA(deg)','DEC(dms)','DEC(deg)']
df = pd.DataFrame(outArray)
df.columns = labels

outLatex = df.to_latex(index=False)
print df.to_latex(index=False)


# In[ ]:

text_file = open("starsLatex.txt", "w")
text_file.write(outLatex)
text_file.close()


# In[ ]:

obsMatrix #stars, observations, [pivot, plate], cam
for i in range(10):
    plt.plot(obsMatrix[i,:,0,0],'.')
plt.show()


# In[ ]:

np.save('obsMatrix.npy',obsMatrix)


# In[ ]:

obsMatrix = np.load('obsMatrix.npy')


# In[ ]:

obsMatrix.shape[0]


# In[ ]:

#stars with consistent plate and fibre
totalStars = 0

for thisStar in range(obsMatrix.shape[0]):
    thisStarsObs = obsMatrix[thisStar,:,:,0]
    if (np.sum(-np.isnan(np.unique(thisStarsObs[:,0])))==1):
        if (np.sum(-np.isnan(np.unique(thisStarsObs[:,1])))==1):
            totalStars += 1
            print thisStar, totalStars


# In[ ]:

for thisStar in range(obsMatrix.shape[0]):
    for thisObs in range(obsMatrix.shape[1]):
        for thisCam in range(obsMatrix.shape[3]):
            if not np.isnan(obsMatrix[thisStar, thisObs, 0, thisCam]):
                FirstObs = [thisStar, thisObs, obsMatrix[thisStar, thisObs, 0, thisCam], obsMatrix[thisStar, thisObs, 1, thisCam], thisCam]
                


# In[ ]:

obsGroup = 0
for thisObs in range(obsMatrix.shape[1]):
    


# ### CC pyhermes

# In[ ]:

cd ~/Documents/HERMES/reductions


# In[ ]:

#HD285507

#cam1
# filep1 = pf.open('pyhermes/HD285507/cam1/1408200037_FIB228_1.fits')
# filep2 = pf.open('pyhermes/HD285507/cam1/1408210036_FIB228_1.fits')
# filep3 = pf.open('pyhermes/HD285507/cam1/1408220039_FIB228_1.fits')
# filep4 = pf.open('pyhermes/HD285507/cam1/1408240063_FIB228_1.fits')
# filep5 = pf.open('pyhermes/HD285507/cam1/1408250047_FIB228_1.fits')

#cam2
# filep1 = pf.open('pyhermes/HD285507/cam2/1408200037_FIB228_2.fits')
# filep2 = pf.open('pyhermes/HD285507/cam2/1408210036_FIB228_2.fits')
# filep3 = pf.open('pyhermes/HD285507/cam2/1408220039_FIB228_2.fits')
# filep4 = pf.open('pyhermes/HD285507/cam2/1408240063_FIB228_2.fits')
# filep5 = pf.open('pyhermes/HD285507/cam2/1408250047_FIB228_2.fits')

#cam3
filep1 = pf.open('pyhermes/HD285507/cam3/1408200037_FIB228_3.fits')
filep2 = pf.open('pyhermes/HD285507/cam3/1408210036_FIB228_3.fits')
filep3 = pf.open('pyhermes/HD285507/cam3/1408220039_FIB228_3.fits')
filep4 = pf.open('pyhermes/HD285507/cam3/1408240063_FIB228_3.fits')
filep5 = pf.open('pyhermes/HD285507/cam3/1408250047_FIB228_3.fits')

filep1_wl = RVT.extract_pyhermes_wavelength(filep1.filename())
filep2_wl = RVT.extract_pyhermes_wavelength(filep2.filename())
filep3_wl = RVT.extract_pyhermes_wavelength(filep3.filename())
filep4_wl = RVT.extract_pyhermes_wavelength(filep4.filename())
filep5_wl = RVT.extract_pyhermes_wavelength(filep5.filename())

filep1_data = filep1[0].data
filep2_data = filep2[0].data
filep3_data = filep3[0].data
filep4_data = filep4[0].data
filep5_data = filep5[0].data


# In[ ]:

#HD1581

#cam1
# filep1 = pf.open('pyhermes/HD1581/cam1/1408200042_FIB176_1.fits')
# filep2 = pf.open('pyhermes/HD1581/cam1/1408210041_FIB176_1.fits')
# filep3 = pf.open('pyhermes/HD1581/cam1/1408220031_FIB176_1.fits')
# filep4 = pf.open('pyhermes/HD1581/cam1/1408240053_FIB176_1.fits')
# filep5 = pf.open('pyhermes/HD1581/cam1/1408250039_FIB176_1.fits')

#cam2
# filep1 = pf.open('pyhermes/HD1581/cam2/1408200042_FIB176_2.fits')
# filep2 = pf.open('pyhermes/HD1581/cam2/1408210041_FIB176_2.fits')
# filep3 = pf.open('pyhermes/HD1581/cam2/1408220031_FIB176_2.fits')
# filep4 = pf.open('pyhermes/HD1581/cam2/1408240053_FIB176_2.fits')
# filep5 = pf.open('pyhermes/HD1581/cam2/1408250039_FIB176_2.fits')

#cam3
# filep1 = pf.open('pyhermes/HD1581/cam3/1408200042_FIB176_3.fits')
filep2 = pf.open('pyhermes/HD1581/cam3/1408210041_FIB176_3.fits')
filep3 = pf.open('pyhermes/HD1581/cam3/1408220031_FIB176_3.fits')
filep4 = pf.open('pyhermes/HD1581/cam3/1408240053_FIB176_3.fits')
filep5 = pf.open('pyhermes/HD1581/cam3/1408250039_FIB176_3.fits')

# filep1_wl = RVT.extract_pyhermes_wavelength(filep1.filename())
filep2_wl = RVT.extract_pyhermes_wavelength(filep2.filename())
filep3_wl = RVT.extract_pyhermes_wavelength(filep3.filename())
filep4_wl = RVT.extract_pyhermes_wavelength(filep4.filename())
filep5_wl = RVT.extract_pyhermes_wavelength(filep5.filename())

# filep1_data = filep1[0].data
filep2_data = filep2[0].data
filep3_data = filep3[0].data
filep4_data = filep4[0].data
filep5_data = filep5[0].data


# In[ ]:

plt.plot(filep1_wl, filep1_data)
plt.plot(filep2_wl, filep2_data)
plt.plot(filep3_wl, filep3_data)
plt.plot(filep4_wl, filep4_data)
plt.plot(filep5_wl, filep5_data)
plt.show()


# In[ ]:

print np.sum(filep1_wl-filep2_wl)
print np.sum(filep1_wl-filep3_wl)
print np.sum(filep1_wl-filep4_wl)
print np.sum(filep1_wl-filep5_wl)


# In[ ]:

reload(RVT)
xDef = 2
corrHWidth=10
CCCurve = []

lambda1, flux1 = RVT.clean_flux(filep1_wl, filep1_data, medianRange=5, xDef=xDef)
# lambda2, flux2 = RVT.clean_flux(filep1_wl, filep1_data, medianRange=5, xDef=xDef)
# lambda2, flux2 = RVT.clean_flux(filep2_wl, filep2_data, medianRange=5, xDef=xDef)
# lambda2, flux2 = RVT.clean_flux(filep3_wl, filep3_data, medianRange=5, xDef=xDef)
# lambda2, flux2 = RVT.clean_flux(filep4_wl, filep4_data, medianRange=5, xDef=xDef)
lambda2, flux2 = RVT.clean_flux(filep5_wl, filep5_data, medianRange=5, xDef=xDef)

# plt.plot(lambda1, flux1)
# plt.plot(lambda2, flux2)
# plt.show()

CCCurve = RVT.signal.fftconvolve(flux1, flux2[::-1], mode='same')
corrMax = np.where(CCCurve==max(CCCurve))[0][0]
p_guess = [corrMax,corrHWidth]
x_mask = np.arange(corrMax-corrHWidth, corrMax+corrHWidth+1)

# plt.plot(CCCurve)
# plt.plot(RVT.gaussian(x_mask, p[0],p[1]))
# plt.show()



p = RVT.fit_gaussian(p_guess, CCCurve[x_mask], np.arange(len(CCCurve))[x_mask])[0]
if np.modf(CCCurve.shape[0]/2.0)[0]>1e-5:
    pixelShift = (p[0]-(CCCurve.shape[0]-1)/2.) #odd number of elements
else:
    pixelShift = (p[0]-(CCCurve.shape[0])/2.) #even number of elements


mid_px = filep2_wl.shape[0]/2
dWl = (filep2_wl[mid_px+1]-filep2_wl[mid_px]) / filep2_wl[mid_px]/xDef
RV = dWl * pixelShift * RVT.constants.c 

print filep2_wl.shape[0], filep2_wl.shape[0]/2, p/xDef
print dWl, pixelShift, RVT.constants.c, pixelShift * RVT.constants.c
print 'RV',RV


# In[ ]:

good results HD285507
cam1
RV 0.650081706571
RV 237.872651165
RV 96.3944140755
RV 114.433467563
RV 50.2164872274

cam2
RV -0.0172313673699
RV -110.892585234
RV 15.6701485047
RV -88.065656879
RV -309.392285134

cam3
RV 0.417713480503
RV 567.479254119
RV 419.397288864
RV -257.445290527
RV -1305.97883204


# In[ ]:

good results HD1581
cam1
RV -0.00651678301497
RV 161.809695302
RV -64.8642306721
RV -56.3130184779


cam2
RV -0.314204070072
RV 174.575284795
RV 101.925175583
RV -63.3004739971


cam3
RV -0.139146305633
RV -94.3366663426
RV -102.202385204
RV -1950.38111841


