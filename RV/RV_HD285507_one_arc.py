# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob
import os
import numpy as np
import scipy.constants as const
from scipy import signal, interpolate, optimize
import pylab as plt
import pickle

import TableBrowser as tb
import RVTools as RVT
reload(RVT)
import create_obj as cr_obj

# <codecell>

os.chdir('/Users/Carlos/Documents/HERMES/reductions/HD285507_single_arc_5.76/uncombined/')

# <codecell>

#load all star names from 1st file
fileList = glob.glob('cam1/*.fits')
a = tb.FibreTable(fileList[0])
starNames=a.target[a.type=='P']

print len(starNames)

# <codecell>

#Loop to create all stars from initial data. No calculations here.
#uses create_obj.py to create stars with basic reduced data
print 'About to create',len(starNames),'stars'

for i in starNames:
    thisStar = cr_obj.star(i)
    thisStar.exposures = cr_obj.exposures()
    thisStar.exposures.load_exposures(thisStar.name)
    thisStar.exposures.calculate_baryVels(thisStar)
    file_pi = open(i+'.obj', 'w') 
    pickle.dump(thisStar, file_pi) 
    file_pi.close()

# <codecell>

#takes each <star_name>.obj and calculates RVs, etc. Saves red_<star_name>.obj
#it won't create the .obj file when errors found, only safe stars will exist but 
#log should be checked to catch good stars that may have been flagged mistakenly.


xDef = 10 #resampling points per pixel
CCMaskWidth = 5 #half width around the peak of the cc curve to be gaussian fitted

for i in starNames:
    print i
    filename = i+'.obj'
    filehandler = open(filename, 'r')
    thisStar = pickle.load(filehandler)
    try:
        RVT.find_max_wl_range(thisStar)
        RVT.RVs_CC_t0(thisStar)

        file_pi = open('red_'+thisStar.name+'.obj', 'w') 
        pickle.dump(thisStar, file_pi) 
        file_pi.close()
    except:
        print 'CC failed (probably NaNs)'
    thisStar = None
    print ''

# <codecell>

# Collects information on all stars and writes them into these arrays:
#     data[thisStar.name, thisStar.Vmag, thisStar.exposures.pivots]
#     RVs
#     sigmas
#     JDs
# It will skip stars with sigma=0.
    
    
fileList = glob.glob('red*.obj')

data = []
RVs = np.zeros((len(fileList),15,4))
sigmas = np.ones((len(fileList),15,4)) * 100

for i in range(len(fileList)):

    print i,fileList[i],
    filehandler = open(fileList[i], 'r') 
    thisStar = pickle.load(filehandler) 
    if ((thisStar.exposures.cameras[0].sigmas.all()!=0) and
        (thisStar.exposures.cameras[1].sigmas.all()!=0) and
        (thisStar.exposures.cameras[2].sigmas.all()!=0) and
        (thisStar.exposures.cameras[3].sigmas.all()!=0)):
        
        data.append([thisStar.name, thisStar.Vmag,np.unique(thisStar.exposures.pivots)[0]])
        RVs[i,:,0] = thisStar.exposures.cameras[0].RVs
        RVs[i,:,1] = thisStar.exposures.cameras[1].RVs
        RVs[i,:,2] = thisStar.exposures.cameras[2].RVs
        RVs[i,:,3] = thisStar.exposures.cameras[3].RVs
        sigmas[i,:,0] = thisStar.exposures.cameras[0].sigmas
        sigmas[i,:,1] = thisStar.exposures.cameras[1].sigmas
        sigmas[i,:,2] = thisStar.exposures.cameras[2].sigmas
        sigmas[i,:,3] = thisStar.exposures.cameras[3].sigmas
        JDs = thisStar.exposures.JDs
        filehandler.close()
        thisStar = None
    else:
        print 'Sigma=0'

    print ''
    
data = np.array(data)
order = np.argsort(data[:,2].astype(int))

data = data[order]
RVs = RVs[order]
sigmas = sigmas[order]

print ''
print 'data',len(data)
print 'RVs',RVs.shape
print 'sigmas',sigmas.shape
print 'JDs',JDs.shape


# <codecell>

# #save?
# np.save('data',data)
# np.save('RVs',RVs)
# np.save('sigmas',sigmas)
# np.save('JDs',JDs)


# <headingcell level=3>

# Data post-process 

# <codecell>

#Load?
data=np.load('data.npy')
RVs=np.load('RVs.npy')
sigmas=np.load('sigmas.npy')
JDs=np.load('JDs.npy')

# <codecell>

#index value of main star in data array
myTarget=20
print data[myTarget] #print to confirm [name, mag, pivot]

# <codecell>

#the full loop

#array to map pivot# to y-position
p2y = RVT.pivot_to_y('20aug10037tlm.fits')

#quad reduced RVs array initialise
quadRVs = np.zeros(RVs.shape)


for thisTarget in range(RVs.shape[0]): #1 loop per target

    thisSigmas = sigmas.copy()
    thisSigmas[thisTarget,:,:] = 1e4 #increase sigma of thisTarget to minimise resulting weight

    #distances to target (in y-coords)
    deltay = p2y[data[:,2].astype(int)] - p2y[data[myTarget,2].astype(int)]
#     mask = np.zeros(RVs.shape[0]).astype(bool)
#     mask[myTarget] = True
#     deltay_mx = np.ma.masked_array(deltay, mask=mask) #removed myTarget
    
    #create RV mask to exclude target and stars with RV>3000m/s\
#     mask = np.zeros(RVs.shape).astype(bool)
#     RVs_mx = np.ma.masked_array(RVs, mask=mask)
#     sigmas_mx = np.ma.masked_array(RVs, mask=mask)
    #     mask[np.unique(np.where(np.abs(RVs)>3000)[0])]=False
#     mask[[33]]=False
    
    c = ['b','g','r','cyan']

    for cam in range(4):
        
        weights = RVT.calibrator_weights(deltay, thisSigmas[:,0,cam])

        for epoch in range(RVs.shape[1]):
            
            order = np.argsort(deltay)
            quadX = deltay[order]
            quadY = RVs[:,epoch,cam][order]
            fRVs,__ = RVT.fit_quad([1,1,0], quadX, quadY )
            fittedCurve = RVT.quad(quadX, fRVs[0], fRVs[1], fRVs[2])
            quadRVs[thisTarget,epoch,cam]=RVs[thisTarget,epoch,cam]-np.sum(weights[order]*(RVs[:,epoch,cam][order])) #-fittedCurve
            if cam == 0:
                if epoch==0:
                    print ''
                    print data[thisTarget,0]
#                     plt.scatter(quadX, quadY, marker= '+', c= 'k' , label='Original RVs')
#                     plt.plot(quadX,fittedCurve, label='Quadratic fit')
#                     plt.scatter(quadX,quadY-fittedCurve, c=c[cam], label='Quadratic corrected RVs')
#                     plt.legend(loc=0)
#                     plt.show()
                print '','RV',RVs[thisTarget,epoch,cam], 'Correction', np.sum(weights[order]*(RVs[:,epoch,cam][order]))
                
#             print RVs[myTarget,epoch,cam]-np.sum(weights*RVs[:,epoch,cam][mask]),RVs[myTarget,epoch,cam],np.sum(weights*RVs[:,epoch,cam][mask])
#             if cam==0:plt.errorbar(epoch,RVs[myTarget,epoch,cam]-np.sum(weights*RVs[:,epoch,cam][mask]))
#             if epoch==3:
# #                 plt.scatter(deltay[mask][order],
# #                             RVs[:,epoch,cam][mask][order]-fittedCurve[myTarget],
# #                             s = 10**3*weights,
# #                             c= 'r', 
# #                             label = cam,
# #                             marker ='o')
#                 plt.scatter(deltay[mask],
#                             RVs[:,epoch,cam][mask],
#                             s = 10**3*weights,
#                             c= c[cam], 
#                             label = cam,
#                             marker ='o')
#                 plt.scatter(deltay[mask], quadRVs[:,epoch,cam][mask], s = 10**3*weights,c= c[cam], label = cam)
#                 plt.scatter(deltay[mask], stableRVs[:,epoch,cam][mask], s = 10**3*weights, marker= '+', c= 'k')
#             order = np.argsort(deltay[mask])
#             plt.plot(deltay[mask][order], (weights*2000)[order], label=i,c= c[cam])
#             plt.plot(deltay[mask][order], (sigmas[:,epoch,cam]*100)[order], label=i,c= c[cam])
#     plt.xticks(data[:,0])
#     ax.set_xticks(deltay[mask])
#     ax.set_xticklabels(data[:,0][mask])
# plt.legend()
# plt.ylabel('RV[m/s]')
# plt.xlabel('deltay[px]')
# plt.show()
quadRVs = np.array(quadRVs)

# <codecell>

plt.scatter(JDs, RVs[myTarget,:,0])
# plt.scatter(JDs, RVs[myTarget,:,0] - thisStar.exposures.rel_baryVels, c='r')
# plt.plot(JDs,thisStar.exposures.rel_baryVels)
plt.scatter(JDs, quadRVs[myTarget,:,0], c='r')
plt.show()

# <codecell>

#array to map pivot# to y-position
p2y = RVT.pivot_to_y('20aug10037tlm.fits')

#create deltay 2D array of distances between fibres
deltay = np.zeros(np.array(RVs.shape)[[0,0]])
for thisTarget in range(RVs.shape[0]):
    deltay[thisTarget,:] = p2y[data[:,2].astype(int)] - p2y[data[thisTarget,2].astype(int)]

# <codecell>

#create 3D weight array

weights = np.zeros(np.array(RVs.shape)[[0,0,2]]) # Same array for all epochs hence epoch-dimention skipped


for thisTarget in range(RVs.shape[0]): #1 loop per target

    #distances to target (in y-coords)
    mask = np.zeros(np.array(RVs.shape)[[0,0]])
    mask[:,thisTarget] = True
    deltay_mx = np.ma.masked_array(deltay, mask=mask)
    
    #create RV mask to exclude target and stars with RV>3000m/s\
    mask = np.zeros(np.array(RVs.shape)[[0,2]])
    mask[thisTarget] = True
    sigmas_1epc = sigmas[:,0,:]
    sigmas_mx = np.ma.masked_array(sigmas_1epc, mask=mask)
    
    c = ['b','g','r','cyan']

    for cam in range(4):
        a = calibrator_weights(deltay_mx[thisTarget,:].compressed(), sigmas_mx[:,cam].compressed())
        weights[thisTarget,:,cam] = np.insert(a, thisTarget, 0)

# <codecell>

#quad reduced RVs array initialise
quadRVs = np.zeros(RVs.shape) 
diffs = np.zeros(RVs.shape) 

mask = np.zeros(RVs.shape).astype(bool)
RVs_mx = np.ma.masked_array(RVs, mask=mask)

for thisTarget in range(RVs.shape[0]): #1 loop per target
    for epoch in range(RVs.shape[1]):
        for cam in range(RVs.shape[2]):
            
            order = np.argsort(deltay_mx[thisTarget,:])
            quadX = deltay_mx[thisTarget,:][order]
            quadY = RVs_mx[:,epoch,cam][order]
            fRVs,__ = fit_quad([1,1,0], quadX, quadY )
            fittedCurve = quad(quadX, fRVs[0], fRVs[1], fRVs[2])
            
            quadRVs[thisTarget,epoch,cam]=RVs[thisTarget,epoch,cam]-np.sum(weights[thisTarget,:,cam][order]*(RVs[:,epoch,cam][order]-fittedCurve))
            diffs[thisTarget,epoch,cam]=np.sum(weights[thisTarget,:,cam][order]*(RVs[:,epoch,cam][order]-fittedCurve))

quadRVs = np.array(quadRVs)
diffs = np.array(diffs)


# <codecell>

# save?
# np.save('quadRVs',quadRVs)

# <headingcell level=3>

# Plots, sanity checks, results

# <codecell>

#delta, sigmas and weights
for thisTarget in range(RVs.shape[0]):
    plt.plot(deltay[0,:], sigmas[0,:,0])
#     plt.plot(deltay[thisTarget,:], weights[thisTarget,:,0]*np.max(sigmas[:,0,0]))
plt.show()

# <codecell>

# quadRVs=np.load('quadRVs.npy')

# <codecell>

fileList[36]
filehandler = open(fileList[36], 'r')
thisStar = pickle.load(filehandler) 

# <codecell>

thisCam=thisStar.exposures.cameras[3]

# <codecell>

for i in range(5):
    plt.plot(thisCam.clean_fluxes[i], label = str(i))
plt.legend(loc=0)
plt.show()

# <codecell>

for i in range(RVs.shape[0]):
    a = RVs.copy()
    a[36,:,0]=np.nan
    # print a
    # print np.nanmean(a[:,:,0], axis=0)
    plt.scatter(JDs, RVs[i,:,1])
    # plt.scatter(JDs, quadRVs[36,:,0], c='r')
plt.show()

# <codecell>

deltay = p2y[data[:,2].astype(int)] - p2y[data[myTarget,2].astype(int)]
mask = np.zeros(RVs.shape[0]).astype(bool)
mask[myTarget] = True
deltay_mx = np.ma.masked_array(deltay, mask=mask)

#create RV mask to exclude target and stars with RV>3000m/s\
mask = np.zeros(RVs.shape).astype(bool)
RVs_mx = np.ma.masked_array(RVs, mask=mask)
sigmas_mx = np.ma.masked_array(sigmas, mask=mask)
sigmas_mx.mask[myTarget,:,:] = True
# print calibrator_weights(deltay_mx.compressed(), sigmas_mx[:,0,cam].compressed()).shape
a = calibrator_weights(deltay_mx, sigmas_mx[:,0,cam])
a = np.insert(a, myTarget, 0)
plt.plot( a)
plt.show()

# <codecell>

quadRVs

# <codecell>

plt.scatter (JDs, quadRVs[36,:,1])
plt.show()

# <codecell>

mask

# <codecell>

fRVs,__ = optimize.curve_fit(quad, quadX, quadY, p0 = [-0.001,-0.001,quadY[np.where(deltay==np.min(np.abs(deltay)))[0][0]]], )
plt.scatter( quadX, quadY)
smoothX = np.linspace(np.min(quadX), np.max(quadX))
fittedCurve = quad(quadX, fRVs[0], fRVs[1], fRVs[2])
plt.plot(quadX,fittedCurve)
plt.scatter(quadX,quadY*fittedCurve, c='r')
plt.show()
print 'params',fRVs

# <codecell>

plt.scatter(JDs,quadRVs[30,:,0], c='r', label = 'stable star (observed)')
plt.scatter(JDs,quadRVs[36,:,1], c='g', label = 'HD285507 (observed)')
start_day = 2456889.500000 # The Julian date for CE  2014 August 20 00:00:00.0 UT  (10am australia)
end_day = 2456895.500000 #The Julian date for CE  2014 August 26 00:00:00.0 UT (10am australia)

days = np.linspace(start_day, end_day)  - 2400000

P = 6.0881
peri_arg = 182
peri_time = 2456257.5- 2400000
K1 =125.8
RV = K1* np.sin((days-peri_time)/P*2*np.pi + peri_arg/360*2*np.pi )
plt.plot(days, RV, linewidth = 1, label = 'HD285507' )
plt.legend(loc=0)
plt.xlabel('JD')
plt.ylabel('RV [m/s]')
plt.show()

# <codecell>

# for i in range(quadRVs.shape[0]):
print np.min((np.nansum(np.abs(quadRVs[:,:,:-1]), axis=1)/4))
print (np.nansum(np.abs(quadRVs), axis=1)/4)


# <codecell>

plt.plot(sigmas.flatten())
plt.show()

# <headingcell level=2>

# Plots

# <codecell>

def plot_all_spec_cams(thisStar):    #All spectra for all cameras
    
    fig, ax = plt.subplots(2,2, sharey='all')
    
    # ax.set_yticks(thisStar.exposures.JDs)
    # ax.set_ylim(np.min(thisStar.exposures.JDs)-1,np.min(thisStar.exposures.JDs)+1)
    for cam in range(4):
        thisCam = thisStar.exposures.cameras[cam]
        fileNames =  thisCam.fileNames
        nFluxes = thisCam.wavelengths.shape[0]
        ax[0,0].set_yticks(np.arange(0,nFluxes))
        ax[0,0].set_ylim(-1,nFluxes)
    
        for i in np.arange(nFluxes):
            d, f = thisCam.clean_wavelengths[i], thisCam.clean_fluxes[i]
            if cam ==0:
                ax[0,0].plot(d, f+i, 'b')
            elif cam==1:
                ax[0,1].plot(d, f+i, 'g')
            elif cam==2:
                ax[1,0].plot(d, f+i, 'r')
            elif cam==3:
                ax[1,1].plot(d, f+i, 'cyan')
        #         ax.plot(d, f+thisStar.exposures.JDs[i], 'k')
    
    plt.xlabel('Wavelength [Ang]')
    plt.title(thisStar.name+' - Camera '+str(cam+1))
    ax[0,0].set_yticklabels(fileNames)
    plt.show()

# <codecell>

RVs1 = np.load('RVs1.npy') 
RVs2 = np.load('RVs2.npy') 
RVs3 = np.load('RVs3.npy') 
RVs4 = np.load('RVs4.npy') 
JDs = np.load('JDs.npy') 

# <codecell>

#Plots RVs, baryvels for all 4 cameras
plt.title('Average of all decorrelated targets')
mask = np.abs(quadRVs)<3000
quadRVs[-mask]=np.nan
# for i in range(RVs.shape[0]):
    # plt.plot(thisStar.exposures.JDs, thisStar.exposures.rel_baryVels, label = 'Barycentric Vel. ')
try:
#         plt.scatter(JDs, stableRVs[i,:,0], label = 'Blue', color ='b' )
#     plt.scatter(JDs, np.median(quadRVs[:,:,0],axis=0), label = 'Blue', color ='b' )
    plt.errorbar(JDs, np.nanmean(quadRVs[:,:,0],axis=0), yerr=np.nanstd(quadRVs[:,:,0],axis=0), label = 'Blue', color ='b' )
#     plt.errorbar(JDs, np.nanmean(RVs[:,:,0],axis=0), yerr=np.nanstd(RVs[:,:,0],axis=0), label = 'Blue', color ='b' )
    pass
except:pass
try:
#     plt.scatter(JDs, stableRVs[i,:,1], label = 'Green', color ='g' )
#     plt.scatter(JDs, np.median(quadRVs[:,:,1],axis=0), label = 'Green', color ='g' )
    plt.errorbar(JDs, np.nanmean(quadRVs[:,:,1],axis=0), yerr=np.nanstd(quadRVs[:,:,1],axis=0), label = 'Green', color ='g' )
#     plt.errorbar(JDs, np.nanmean(RVs[:,:,1],axis=0), yerr=np.nanstd(RVs[:,:,1],axis=0), label = 'Green', color ='g' )
    pass
except:pass
try:
#     plt.scatter(JDs, stableRVs[i,:,2], label = 'Red', color ='r' )
#     plt.scatter(JDs, np.median(quadRVs[:,:,2],axis=0), label = 'Red', color ='r' )
    plt.errorbar(JDs, np.nanmean(quadRVs[:,:,2],axis=0), yerr=np.nanstd(quadRVs[:,:,2],axis=0), label = 'Red', color ='r' )
#     plt.errorbar(JDs, np.nanmean(RVs[:,:,2],axis=0), yerr=np.nanstd(RVs[:,:,2],axis=0), label = 'Red', color ='r' )
    pass
except:pass
try:
#     plt.scatter(JDs, stableRVs[i,:,3], label = 'IR', color ='cyan' )
#     plt.scatter(JDs, np.median(stableRVs[:,:,3],axis=0), label = 'IR', color ='cyan' )
    plt.errorbar(JDs, np.nanmean(quadRVs[:,:,3],axis=0), yerr=np.nanstd(quadRVs[:,:,3],axis=0), label = 'IR', color ='cyan' )
    pass
except:pass

# start_day = 56889.000000 # The Julian date for CE  2014 August 20 00:00:00.0 UT  (10am australia)
# end_day = 56895.000000 #The Julian date for CE  2014 August 26 00:00:00.0 UT (10am australia)

# days = np.linspace(start_day, end_day) 

# K1=26100
# peri_time = 19298.85
# P=4.8202
# peri_arg=269.3
# RV = K1* np.sin( (days-peri_time)/P*2*np.pi - peri_arg/360*2*np.pi )
# plt.plot(days, RV, linewidth = 1, label = 'rhoTuc' )

plt.xlabel('JD')
plt.ylabel('RV [m/s]')
plt.legend(loc=0)
plt.show()

# <codecell>

mask = np.abs(quadRVs)<3000
plt.plot(quadRVs[mask].flatten())
plt.show()

# <codecell>

# np.save('RVs1',RVs1) 
# np.save('RVs2',RVs2) 
# np.save('RVs3',RVs3) 
# np.save('RVs4',RVs4) 
# np.save('JDs',JDs) 

# <codecell>

#Plots RVs, baryvels. Single star, 4 cameras
plt.title(filehandler.name[:-4])
# plt.plot(thisStar.exposures.JDs, thisStar.exposures.rel_baryVels, label = 'Barycentric Vel. ')

thisCam = thisStar.exposures.cameras[0]
# RVMask = thisStar.exposures.my_data_mask
# RVMask = ((thisStar.exposures.my_data_mask) & (np.abs(thisCam.RVs)<50000))
RVMask = thisCam.safe_flag
plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'Blue', color ='b' )
# plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'Blue', color ='b' )

thisCam = thisStar.exposures.cameras[1]
# RVMask = thisStar.exposures.my_data_mask
# RVMask = np.abs(thisCam.RVs)<1e6
RVMask = thisCam.safe_flag
plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'Green' , color ='g')
# plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.RVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'Green' , color ='g')

thisCam = thisStar.exposures.cameras[2]
# RVMask = thisStar.exposures.my_data_mask
# RVMask = np.abs(thisCam.RVs)<2e5
RVMask = thisCam.safe_flag
plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'Red' , color ='r')
# plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.RVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'Red' , color ='r')

thisCam = thisStar.exposures.cameras[3]
# RVMask = thisStar.exposures.my_data_mask
# RVMask = ((thisStar.exposures.my_data_mask) & (np.abs(thisCam.RVs)<20000))
RVMask = thisCam.safe_flag
plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'IR', color ='cyan' )
# plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.RVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'IR', color ='cyan' )

# start_day = 56889.000000 # The Julian date for CE  2014 August 20 00:00:00.0 UT  (10am australia)
# end_day = 56895.000000 #The Julian date for CE  2014 August 26 00:00:00.0 UT (10am australia)

# days = np.linspace(start_day, end_day) 

# K1=26100
# peri_time = 19298.85
# P=4.8202
# peri_arg=269.3
# RV = K1* np.sin( (days-peri_time)/P*2*np.pi - peri_arg/360*2*np.pi )
# plt.plot(days, RV, linewidth = 1, label = 'rhoTuc' )

plt.xlabel('JD')
plt.ylabel('RV [m/s]')
plt.legend(loc=3)
plt.show()

# <codecell>

thisCam = thisStar.exposures.cameras[0]
# for i in range(5):
print thisCam.Ps
print (thisCam.Ps[0,0]-thisCam.clean_wavelengths[0].shape[0]/2)
print (-20480)*3000

# <codecell>

#plots all data
thisCam = thisStar.exposures.cameras[2]
for i in [3]:
    
    fig = plt.gcf()
    fig.suptitle(filehandler.name[:-4]+' - t0 vs t'+str(i)+' - RV='+str(thisCam.RVs[i])+' m/s', fontsize=14)

    plt.subplot(221)
    plt.title('Clean Flux')
    plt.plot(thisCam.clean_wavelengths[i][::50],thisCam.clean_fluxes[i][::50])
    plt.xlabel('Wavelength [ang]')
    plt.ylabel('Flux [Counts]')
    plt.legend(loc=0)
    
    plt.subplot(223)
    plt.title('Clean Flux - Detail')
    plt.plot(thisCam.clean_wavelengths[0],thisCam.clean_fluxes[0], label = 't0 Flux')
    plt.plot(thisCam.clean_wavelengths[i],thisCam.clean_fluxes[i], label = 'Epoch '+str(i))
#     plt.axis((4859,4864, -1,1))
    plt.xlabel('Wavelength [ang]')
    plt.ylabel('Flux [Counts]')
    plt.legend(loc=0)

    plt.subplot(222)
    plt.title('Cross Correlation Result')
    plt.plot((thisCam.clean_wavelengths[i]-np.median(thisCam.clean_wavelengths[i]))/thisCam.clean_wavelengths[i]*const.c,thisCam.CCCurves[i]/np.max(thisCam.CCCurves[i]))
    plt.plot((thisCam.clean_wavelengths[i]-np.median(thisCam.clean_wavelengths[i]))/thisCam.clean_wavelengths[i]*const.c,
             gaussian(range(len(thisCam.clean_wavelengths[i])), thisCam.Ps[i][0], thisCam.Ps[i][1]), label = 'gaussian fit') 
    plt.xlabel('RV [m/s]')
    plt.legend(loc=0)
    
    plt.subplot(224)
    plt.title('Cross Correlation Result - Detail')
    plt.plot((thisCam.clean_wavelengths[i]-np.median(thisCam.clean_wavelengths[i]))/thisCam.clean_wavelengths[i]*const.c,thisCam.CCCurves[i]/np.max(thisCam.CCCurves[i]))
    plt.plot((thisCam.clean_wavelengths[i]-np.median(thisCam.clean_wavelengths[i]))/thisCam.clean_wavelengths[i]*const.c,
             gaussian(range(len(thisCam.clean_wavelengths[i])), thisCam.Ps[i][0], thisCam.Ps[i][1]), label = 'gaussian fit') 
    plt.axis((-8000, 8000,0.5,1.1) )
    plt.xlabel('RV [m/s]')
    plt.legend(loc=2)    

    plt.tight_layout()
    plt.show()

# <codecell>


# <codecell>

#plots all CC fitted gausian results\n",
# for i in range(CCCurves.shape[0]):\n",
thisCam = thisStar.exposures.cameras[0]
for i in range(10):
    plt.plot((thisCam.clean_wavelengths[i]-np.median(thisCam.clean_wavelengths[i]))/thisCam.clean_wavelengths[i]*const.c,thisCam.CCCurves[i]/np.max(thisCam.CCCurves[i]))
#     plt.plot((thisCam.clean_wavelengths[i]-np.median(thisCam.clean_wavelengths[i]))/thisCam.clean_wavelengths[i]*const.c,
#              gaussian(range(len(thisCam.clean_wavelengths[i])), thisCam.Ps[i][0], thisCam.Ps[i][1])) 
    plt.xlabel('RV [m/s]')
plt.show()

# <codecell>

#Debuging CC - resampling and cleaning tests
flux1 = thisStar.exposures.cameras[0].red_fluxes[0]
lambda1 = thisStar.exposures.cameras[0].wavelengths[0]
lambda1Clean_10, flux1Clean_10 = clean_flux(flux1, 1, lambda1)
# lambda1Clean_10, flux1Clean_10 = clean_flux(flux1, xDef, lambda1)
# lambda1Clean_100, flux1Clean_100 = clean_flux(flux1, 100, lambda1)\n",
plt.plot(lambda1,flux1/np.max(flux1), label= 'Reduced')
# plt.plot(lambda1Clean_1,flux1Clean_1)
plt.plot(lambda1Clean_10,flux1Clean_10, label= 'Clean')
# plt.plot(lambda1Clean_100,flux1Clean_100)
plt.title('Reduced and Clean flux')
plt.xlabel('Wavelength [Ang.]')
plt.ylabel('Relatuve Flux')
plt.legend(loc=0)
plt.show()

# <codecell>

#Saves
# delattr(thisStar.exposures.cameras[0],'clean_fluxes')
# delattr(thisStar.exposures.cameras[1],'clean_fluxes')
# delattr(thisStar.exposures.cameras[2],'clean_fluxes')
# delattr(thisStar.exposures.cameras[3],'clean_fluxes')
# delattr(thisStar.exposures.cameras[0],'clean_wavelengths')
# delattr(thisStar.exposures.cameras[1],'clean_wavelengths')
# delattr(thisStar.exposures.cameras[2],'clean_wavelengths')
# delattr(thisStar.exposures.cameras[3],'clean_wavelengths')
# file_pi = open(filehandler.name, 'w') 
# pickle.dump(thisStar, file_pi) 
# file_pi.close()
# filehandler.close()
# thisStar = None

# <codecell>


# <codecell>


