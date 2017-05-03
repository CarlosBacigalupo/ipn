# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pandas as pd
import pylab as plt

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

cd ~/Documents/HERMES/reductions/6.5/HD285507_1arc/npy/

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


# <codecell>

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

# <codecell>

SNRs=np.load('SNRs.npy')
import toolbox as tb

# <codecell>

for i,a in enumerate(zip(data[:,],SNRs[:,0,0])):
    print i,a[0],a[1], tb.dec2sex(float(a[0][3])/15), tb.dec2sex(float(a[0][4]))

# <codecell>

pwd

# <codecell>


