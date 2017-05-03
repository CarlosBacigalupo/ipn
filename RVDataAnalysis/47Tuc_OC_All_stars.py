
# coding: utf-8

# In[1]:

import glob
import pickle
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


# In[2]:

def f1(x):
    return '%5.2f' % x


# In[3]:

cd /Users/Carlos/Documents/HERMES/reductions/6.5/47Tuc_core/obj/


# In[8]:

filehandler = open('Field01.obj', 'r')
thisStar = pickle.load(filehandler)
thisCamera= thisStar.exposures.cameras[0]
thisCamera.fileNames
thisStar.exposures.plates[10][-1]
# print np.vstack((thisStar.exposures.pivots,np.nansum(thisCamera.red_fluxes,axis=1))).transpose()
# print thisStar.exposures.p


# In[5]:


outFileNames = np.load('/Users/Carlos/Documents/ipn/npy/outFileNames.npy')
# outObjects = np.load('npy/outObjects.npy')
# outPlates = np.load('npy/outPlates.npy')
outMJD_Exp = np.load('/Users/Carlos/Documents/ipn/npy/outMJD_Exp.npy')

# outFileNames = np.delete(outFileNames,86)
# outObjects = np.delete(outObjects,86)
# outPlates = np.delete(outPlates,86)
# outMJD_Exp = np.delete(outMJD_Exp,86, axis = 0)

ordIdx = np.argsort(outMJD_Exp[:,0])

outFileNames = outFileNames[ordIdx]
outMJD_Exp = outMJD_Exp[ordIdx]



# In[6]:

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

labels = ['ID', 'Name','Vmag','RA(hms)','RA(dms)','RA(deg)','DEC(dms)','DEC(deg)']
df = pd.DataFrame(outArray)
df.columns = labels

outLatex = df.to_latex(index=False)
print df.to_latex(index=False)


# In[ ]:



