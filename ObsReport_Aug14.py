
# coding: utf-8

# In[38]:




# In[34]:

cd /Users/Carlos/Documents/HERMES/data/140821/data/ccd_1/


# In[67]:

a = pf.open('21aug10030.fits')


# In[23]:

ah = a[1].data


# In[24]:

print ah[ah['NAME']=='Giant01']
print 'Pivot',ah[ah['NAME']=='Giant01']['PIVOT'][0]
print np.where(ah['NAME']=='Giant01')[0][0]


# In[7]:

dirs = ['/Users/Carlos/Documents/HERMES/data/140820/data/ccd_1/',
        '/Users/Carlos/Documents/HERMES/data/140821/data/ccd_1/',
        '/Users/Carlos/Documents/HERMES/data/140822/data/ccd_1/',
        '/Users/Carlos/Documents/HERMES/data/140824/data/ccd_1/',
        '/Users/Carlos/Documents/HERMES/data/140825/data/ccd_1/']


# In[63]:

print a[0].header['UTMJD']
print a[0].header['UTSTART']
print a[0].header['UTDATE']
print a[0].header['EXPOSED']
print 
e = float(a[0].header['EXPOSED'])/2/24/60/60 # EXPOSED/2 in days
inDate = np.hstack((a[0].header['UTDATE'].split(':'),a[0].header['UTSTART'].split(':'))).astype(int)
print toolbox.gd2jd(inDate, TZ=0)-2400000.5 + e


# In[98]:

cd ..


# In[28]:

import pyfits as pf
import numpy as np
import pylab as plt
import os
import glob
import pickle
import RVTools as RVT
import toolbox

raw = []
print 'Obs & Filename & Field Name & Plate & MJD & Relative Day (days) & Exp Time (s) \\\\'
good_targets = np.array(['rhoTuc #176 (Pivot 175)', 'HD1581 #176 (Pivot 175)','HD285507 #227 (Pivot 223)', '47Tuc center', '47Tuc IFU' ])
# good_targets = np.array(['rhoTuc #176 (Pivot 175)', 'HD1581 #176 (Pivot 175)','HD285507 #227 (Pivot 223)' ])
# good_targets = np.array(['HD285507 #227 (Pivot 223)' ])
# good_targets = np.array(['M67 12V14','M67 Bright Stars' ])
# good_targets = np.array(['HD1581 #176 (Pivot 175)'])
# good_targets = np.array(['47Tuc center'])

# good_pivots = [175,175,224]
t0 = 0  
order = 0 
# dirs = glob.glob('*')
# print dirs
for j in dirs:
#     print os.path.abspath(j +'/data/ccd_1/')
#     os.chdir(j+'/data/ccd_1/')
#     print j 
    for i in glob.glob(os.path.abspath(j +'*.fits')):
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
#                 print str(order)+' & '+ i+' & '+a[0].header['OBJECT']+' & '+a[0].header['SOURCE']+' & '+str(UTMJD)+' & '+str(tDiff)+' & '+str(a[0].header['EXPOSED']) +' \\\\'
                print order,
                order +=1
                raw.append((i, a[0].header['OBJECT'],a[0].header['SOURCE'],a[0].header['UTMJD'],a[0].header['EXPOSED']))            
        a.close
# raw = np.array(raw)


# In[73]:

a = np.array([-5,-1,0,5,3])
np.abs(a)>4


# In[45]:

targets = np.unique(np.array(raw)[:,1])
good_targets = ['47Tuc IFU', '47Tuc center','rhoTuc #176 (Pivot 175)', 'HD1581 #176 (Pivot 175)','HD285507 #227 (Pivot 223)' ]
target_alias = ['$\rho$ Tucanae #176', 'HD1581 #176','HD285507 #227', '47 Tucanae core', '47 Tucanae IFU' ]


# In[44]:

raw


# In[58]:

# colors = ['k','k','k','k','k','k','k']
colors = ['g','g','g','g','g','g','g','g','g','g','g','g']
dates = ['20 aug', '21 aug', '22 aug', '23 aug', '24 aug', '25 aug', '']

fig, ax = plt.subplots()
ax.set_yticks(np.arange(1,6))
ax.set_ylim(0,6)
plt.xlabel('date')
plt.title('Observations per target')

idx_y = [5,4,2,3,1,1,1,1]
for i in raw:
    idx = np.where(i[1]==targets)[0][0]
    if targets[idx] in good_targets:
        ax.scatter(i[3], idx_y[idx] ,c = colors[idx], s =1, linewidths=20, marker ='_')

    
ax.set_yticklabels(target_alias)
ax.set_xticklabels(dates)
plt.gca().add_patch(plt.Rectangle((2456890.083333-2400000.5,-100),0.3,200, alpha = 0.2, label = 'Observing Window'))
plt.gca().add_patch(plt.Rectangle((2456891.083333-2400000.5,-100),0.3,200, alpha = 0.2))
plt.gca().add_patch(plt.Rectangle((2456892.083333-2400000.5,-100),0.3,200, alpha = 0.2))
plt.gca().add_patch(plt.Rectangle((2456894.083333-2400000.5,-100),0.3,200, alpha = 0.2))
plt.gca().add_patch(plt.Rectangle((2456895.083333-2400000.5,-100),0.3,200, alpha = 0.2))
plt.legend()
plt.show()


# In[26]:

# retrieve SNR from all files for a selected target
SNR = np.zeros((1,15,4))
expTime = []
JDs = []

for cam in range(1,5):
    t = 0
    for epoch in [1,2,3,3,5]:
        os.chdir('/Users/Carlos/Documents/HERMES/data/14082'+str(epoch)+'/data')
        fileList = glob.glob('ccd_'+str(cam)+'/*.fits')
        for thisFileIdx in range(len(fileList)):
            a = pf.open(fileList[thisFileIdx])
            if a[0].header['OBJECT']=='HD1581 #176 (Pivot 175)':
                flux = a[0].data[176]
#                 print cam, a[0].header['UTMJD'], np.median(flux)/np.std(flux), a[0].header['EXPOSED']
                SNR[0, t, cam-1]=np.median(flux)/np.std(flux)
                if cam ==1: 
                    JDs.append(a[0].header['UTMJD'])
                    expTime.append(a[0].header['EXPOSED'])
                t +=1
JDs = np.array(JDs)
expTime = np.array(expTime)
                


# In[3]:

#opens a single star
filename = '/Users/Carlos/Documents/HERMES/reductions/HD1581_6.0/uncombined/red_Giant01.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)


# In[4]:

thisCam=thisStar.exposures.cameras[0]


# In[5]:

plt.plot(thisCam.red_fluxes[0])
plt.show()


# In[10]:

flux = thisCam.red_fluxes[0]
plt.plot(np.real(np.fft.fft(np.sin(np.arange(1000*2*np.pi)))))
# plt.plot(np.sin(np.arange(1000*2*np.pi)))
plt.show()


# In[8]:

np.real(np.fft.fft(np.sin(np.arange(1000*2*np.pi))))


# In[ ]:

# retrieve SNR from all files for a selected target
SNR = np.zeros((1,15,4))
expTime = []
JDs = []

for cam in range(4):
    thisCam=thisStar.exposures.cameras[cam]
    for epoch in [0,1,2,4,5]:
        os.chdir('/Users/Carlos/Documents/HERMES/data/14082'+str(epoch)+'/data')
        fileList = glob.glob('ccd_'+str(cam)+'/*.fits')
        for thisFileIdx in range(len(fileList)):
            a = pf.open(fileList[thisFileIdx])
            if a[0].header['OBJECT']=='HD1581 #176 (Pivot 175)':
                flux = a[0].data[176]
#                 print cam, a[0].header['UTMJD'], np.median(flux)/np.std(flux), a[0].header['EXPOSED']
                SNR[0, t, cam-1]=np.median(flux)/np.std(flux)
                if cam ==1: 
                    JDs.append(a[0].header['UTMJD'])
                    expTime.append(a[0].header['EXPOSED'])
                t +=1
JDs = np.array(JDs)
expTime = np.array(expTime)
                


# In[11]:

colors = ['b', 'g', 'r', 'cyan']
cameras = ['Blue', 'Green', 'Red', 'IR']
for cam in range(4):
    plt.scatter(JDs, SNR[0,:,cam], s = expTime, c = colors[cam], label = cameras[cam])
plt.legend(loc=0)
plt.ylabel('SNR')
plt.xlabel('JD')
plt.show()


# In[49]:

## Changes the OBJECT header property to reflect offsets and fibre changes - USE WITH CAUTION
os.chdir('/Users/Carlos/Documents/HERMES/data/140825')
front = '25aug'
back = '0052.fits'
#newName = '47Tuc IFU'
#newName = '47Tuc center'
#newName = 'rhoTuc #392 (Pivot 398)'
newName = 'rhoTuc #176 (Pivot 175)'
#newName = 'HD1581 #392 (Pivot 398)'
#newName = 'HD1581 #176 (Pivot 175)'
#newName = 'HD285507 #227 (Pivot 223)'
for i in range(1,5):
    a = pf.open('data/ccd_'+str(i)+'/'+ front + str(i) + back)
    a[0].header['OBJECT']=newName
    a.writeto('data/ccd_'+str(i)+'/'+ front + str(i) + back, clobber=True)  
    a.close()


# In[ ]:



