
# coding: utf-8

# In[8]:

# import os
# import create_obj as cr_obj
# reload(cr_obj)
# import pickle
import glob
import pyfits as pf
# import importlib
import numpy as np
# import sys
import toolbox
import pandas as pd


# In[10]:

cd ~/Documents/HERMES/reductions/6.5/m67_lr/


# In[11]:

#create list of stars, pivot, plate, mjd
starNames = np.array([])
fileList = glob.glob('cam1/*.fits')

for fitsname in fileList[:]:
#     print "Starnames",starNames, 'file', fitsname
    HDUList = pf.open(fitsname)
    
    #plate
    plate = HDUList[0].header['SOURCE'].strip()[-1]
    
    #MJD
    e = float(HDUList[0].header['EXPOSED'])/2/24/60/60 # EXPOSED/2 in days
    inDate = np.hstack((HDUList[0].header['UTDATE'].split(':'),HDUList[0].header['UTSTART'].split(':'))).astype(int)
    MJD = toolbox.gd2jd(inDate, TZ=0) - 2400000.5 + e
    
#     print fitsname, MJD
    
    a = HDUList['FIBRES'].data     
    if len(starNames)==0:
        starNames = np.array([a.field('NAME').strip(),
                              a.field('PIVOT'),
                              np.tile(plate,400),
                              np.tile(MJD,400) ]).transpose()[a.field('TYPE').strip()=='P']
    else:
        starNames = np.vstack((starNames,np.array([a.field('NAME').strip(),
                                   a.field('PIVOT'),
                                   np.tile(plate,400),
                                   np.tile(MJD,400) ]).transpose()[a.field('TYPE').strip()=='P']))
#     starNames = np.hstack((starNames,np.array(a.field('NAME')[a.field('TYPE').strip()=='P'])))
    HDUList.close()

# starNames = np.unique(starNames)
starNames = np.array(starNames)

print starNames.shape, 'stars'
# print starNames
# a.columns


# In[12]:

#Convert starNames to name_pivot_plate,MJD

starNames_piv_pl = np.chararray((starNames.shape[0], 2),itemsize = 40)

for i, line in enumerate(starNames):
    starNames_piv_pl[i,0] = line[0]+'_'+line[1]+'_'+line[2]
    starNames_piv_pl[i,1] = line[3]

print starNames_piv_pl


# In[13]:

#Convert starNames_piv_pl,mjd to unique name_piv_pl vs MJD table

rows = np.unique(starNames_piv_pl[:,0])
cols = np.sort(np.unique(starNames_piv_pl[:,1]))
obsMatrix = np.zeros((rows.shape[0], cols.shape[0]))



for line in starNames_piv_pl:
#     print line,
    rowIdx = np.where(rows==line[0])[0][0]
    colIdx = np.where(cols==line[1])[0][0]
    obsMatrix[rowIdx, colIdx] = 1


# In[14]:

def X(x):
    if int(x)==0:
        result = ''
    else:
        result = 'X'
    return result

def nottin(x):
    return x


# In[15]:

all_MJD_Exp = np.load('/Users/Carlos/Documents/ipn/npy/outMJD_Exp.npy')
all_MJD_Exp = np.array(all_MJD_Exp, dtype='|f8')
ordIdx = np.argsort(all_MJD_Exp[:,0])
allMJDs = all_MJD_Exp[:,0][ordIdx]


# In[16]:

x = np.array(cols, dtype='|f8')
colsFlt = x.astype(np.float)
print colsFlt.shape


# In[17]:

booResult = np.in1d(np.round(allMJDs, 5), np.round(colsFlt,5))
colsIdx = np.arange(allMJDs.shape[0])[booResult]
print np.sum(booResult), booResult. shape, allMJDs.shape, colsFlt.shape, colsIdx.shape


# In[18]:

#print matrix in latex form 

a = pd.DataFrame(obsMatrix)
# a.columns = cols #this for MJDs
a.columns = colsIdx #this for obsID
a.index = rows
print a.to_latex(formatters=[X, X, X, X, X,
                             X, X, X, X, X,
                             X, X, X, X, X,
                             X, X, X, X, X,
                             X, X, X, X, X,
                             X, X, X, X, X,
                             X, X, X, X, X,
                             X, X, X, X, X,
                             X, X, X, X, X,
                             X, X, X, X, X,
                             X, X, X, X, X,
                             X, X, X, X, X,
                             X, X, X, X, X])


# In[32]:

#print matrix in latex form 
page = 3
perPage = 20
thisPage = np.arange(perPage*(page-1),perPage*page)
thisPage = np.arange(60,65)
print thisPage, colsIdx[thisPage]


a = pd.DataFrame(obsMatrix[:,thisPage])
# a.columns = cols #this for MJDs
a.columns = colsIdx[thisPage] #this for obsID
a.index = rows
print a.to_latex(formatters=[X, X, X, X, X,
                             X, X, X, X, X,
                             X, X, X, X, X,
                             X, X, X, X, X,
                             X, X, X, X, X])


# In[ ]:

counter = 0
for i in range(colsFlt.shape[0]):
    print i, colsFlt[i],
    booResult2 = np.round(colsFlt[i], 5)==np.round(allMJDs, 5)
    if np.sum(booResult2)>0:
        counter +=1
        print allMJDs[booResult2], np.sum(booResult2), counter
    else:
        print 


# In[ ]:

counter = 0
for i in range(100):
    print i, allMJDs[i],
    booResult2 = np.round(allMJDs[i], 5)==np.round(colsFlt, 5)
    if np.sum(booResult2)>0:
        counter +=1
        print colsFlt[booResult2], np.sum(booResult2), counter
    else:
        print 


# In[296]:

#create list of first observation MJDs
firstObs = []
for i,thisStarPivPl in enumerate(np.unique(starNames_piv_pl[:,0])):
    thisMJDs = starNames_piv_pl[:,1][np.where(thisStarPivPl==starNames_piv_pl[:,0])[0]]
    x = np.array(thisMJDs, dtype='|f8')
    MJDsFlt = x.astype(np.float)
    firstObs.append(np.min(MJDsFlt))

print len(firstObs)
firstObs = np.unique(firstObs)
print len(firstObs)


# In[330]:

#create name_piv_pl_mjd for all tagets
x = np.array(starNames_piv_pl[:,1], dtype='|f8')
MJDsFlt = x.astype(np.float)

starNames_piv_pl_mjd=[]

for i in firstObs:
    thisSet = starNames_piv_pl[:,0][np.where(i==MJDsFlt)]+'_'+str(i)
    for j in thisSet:
#         print j,
        starNames_piv_pl_mjd.append(j)
    
starNames_piv_pl_mjd = np.array(starNames_piv_pl_mjd)
np.save('starNames_piv_pl_mjd.npy',starNames_piv_pl_mjd)


# In[332]:

#the number of unique name_piv_pl_firstMJD
starNames_piv_pl_mjd.shape


# In[ ]:

#tests


# In[335]:

combinedName = starNames_piv_pl_mjd[0]


# In[337]:

starName, pivot, plate, MJD = combinedName.split('_')


# In[338]:

print starName, pivot, plate, MJD


# In[ ]:



