# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pandas as pd

# <codecell>

def f1(x):
    return '%5.2f' % x

# <headingcell level=3>

# 47tuc

# <codecell>

cd ~/Documents/HERMES/reductions/47Tuc_core_6.2/npy

# <codecell>

cd ~/Documents/HERMES/reductions/47Tuc_core_6.2/npy_150509/

# <codecell>

cd ~/Documents/HERMES/reductions/47Tuc_core_1arc_6.2/npy/

# <codecell>

cd ~/Documents/HERMES/reductions/47Tuc_core_1arc_6.2/npy_150509/

# <headingcell level=3>

# HD1581

# <codecell>

cd ~/Documents/HERMES/reductions/HD1581_6.2/npy/

# <headingcell level=3>

# HD285507

# <codecell>

cd ~/Documents/HERMES/reductions/HD285507_6.2/npy_150509/

# <codecell>

cd ~/Documents/HERMES/reductions/HD285507_6.2/npy/

# <codecell>

cd ~/Documents/HERMES/reductions/HD285507_1arc_6.2/npy_150509/

# <codecell>

cd ~/Documents/HERMES/reductions/HD285507_1arc_6.2/npy/

# <headingcell level=3>

# m67

# <codecell>

cd ~/Documents/HERMES/reductions/m67_6.2/npy

# <codecell>

cd ~/Documents/HERMES/reductions/m67_6.2/npy_raw/

# <codecell>

cd ~/Documents/HERMES/reductions/m67_1arc_6.2/

# <headingcell level=3>

# NGC2477

# <codecell>

cd ~/Documents/HERMES/reductions/NGC2477_6.2/npy

# <codecell>

cd ~/Documents/HERMES/reductions/NGC2477_6.2/npy_150509/

# <codecell>

cd ~/Documents/HERMES/reductions/NGC2477_1arc_6.2/

# <headingcell level=3>

# rhoTuc

# <codecell>

cd ~/Documents/HERMES/reductions/rhoTuc_6.2/npy

# <codecell>

cd ~/Documents/HERMES/reductions/6.2/rhoTuc_1arc_6.2/npy

# <headingcell level=3>

# % of stars within a given RV

# <codecell>

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

# <headingcell level=3>

# Shows all observatios for a list of 'good_targets'

# <codecell>

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
good_targets = np.array(['rhoTuc #176 (Pivot 175)'])
# good_targets = np.array(['HD285507 #227 (Pivot 223)' ])
# good_targets = np.array(['M67 12V14','M67 Bright Stars' ])
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

# <headingcell level=3>

# Stars Names and magnitudes

# <codecell>

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

# <headingcell level=3>

# W tests

# <codecell>


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

# <codecell>


