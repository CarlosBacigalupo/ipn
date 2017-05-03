# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob
import os
import numpy as np
from scipy import signal, interpolate, optimize, constants
import pylab as plt
import pickle
import create_obj as cr_obj
reload(cr_obj)
import RVTools as RVT
reload(RVT)
import pyfits as pf

# <codecell>

#single exposure data reduction

# <codecell>

cd /Users/Carlos/Documents/HERMES/reductions/rhoTuc_6.0/uncombined/

# <codecell>

#uses create_obj.py to create star with basic reduced data
thisStar = cr_obj.star('Giant01')
thisStar.exposures = cr_obj.exposures()
thisStar.exposures.load_exposures(thisStar.name)
thisStar.exposures.calculate_baryVels(thisStar)
thisStar.name = 'rhoTuc'
file_pi = open('rhoTuc.obj', 'w') 
pickle.dump(thisStar, file_pi) 
file_pi.close()

# <codecell>

#takes each <star_name>.obj and calculates RVs, etc. Saves red_<star_name>.obj
reload(RVT)
xDef = 10 #resampling points per pixel
CCMaskWidth = 5 #half width around the peak of the cc curve to be gaussian fitted

filename = 'rhoTuc.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

RVT.find_max_wl_range(thisStar)
RVT.RVs_CC_t0(thisStar)
file_pi = open('red_rhoTuc.obj', 'w') 
pickle.dump(thisStar, file_pi) 
file_pi.close()

# <codecell>

cd /Users/Carlos/Documents/HERMES/reductions/HD1581_6.0/single_exposures/22

# <codecell>

#uses create_obj.py to create star with basic reduced data
thisStar = cr_obj.star('Giant01')
thisStar.exposures = cr_obj.exposures()
thisStar.exposures.load_exposures(thisStar.name)
thisStar.exposures.calculate_baryVels(thisStar)
thisStar.name = 'rhoTuc'
file_pi = open('rhoTuc.obj', 'w') 
pickle.dump(thisStar, file_pi) 
file_pi.close()

# <codecell>

#takes each <star_name>.obj and calculates RVs, etc. Saves red_<star_name>.obj
reload(RVT)
xDef = 10 #resampling points per pixel
CCMaskWidth = 5 #half width around the peak of the cc curve to be gaussian fitted

filename = 'rhoTuc.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

RVT.find_max_wl_range(thisStar)
RVT.RVs_CC_t0(thisStar)
file_pi = open('red_rhoTuc.obj', 'w') 
pickle.dump(thisStar, file_pi) 
file_pi.close()

# <headingcell level=3>

# Plots, sanity checks, results

# <codecell>


Dec1	Dec2	Dec3	No	RA1	RA2	RA3	   SpecType1	Vmag1	K1	K1_P	K2	eccentricity	grade	peri_arg	peri_time	period(days)
-65	     28	    4.91	40	0	42	28.373	F6V	       5.393	26.10	5.414713	NaN	0.02	5	269.3	19299.110	4.820200

# <codecell>

colors = ['b','g','r','cyan']
cameras = ['Blue', 'Green', 'Red', 'IR']

# <codecell>

cd /Users/Carlos/Documents/HERMES/reductions/HD1581_6.0/single_exposures/22

# <codecell>

cd /Users/Carlos/Documents/HERMES/reductions/rhoTuc_6.0/uncombined/

# <codecell>

filename = 'red_rhoTuc.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)
thisCam=thisStar.exposures.cameras[0]
thatCam=thisStar.exposures.cameras[1]

# <codecell>

thisCam=thisStar.exposures.cameras[0]
thatCam=thisStar.exposures.cameras[1]

rv_avg = np.ones(5) * np.nan
rv_std = np.ones(5) * np.nan
JD_avg = np.ones(5) * np.nan
BCV_avg = np.ones(5) * np.nan

arr = np.hstack((thisCam.RVs[0:8],thatCam.RVs[0:8]))
JD_avg[0] = np.average(thisStar.exposures.JDs[0:8])
BCV_avg[0] = np.average(thisStar.exposures.rel_baryVels[0:8])
rv_avg[0] = np.average(arr)
rv_std[0] = np.std(arr)
JD_avg[0] = np.min(thisStar.exposures.JDs)
rv_avg[0] = 0

arr = np.hstack((thisCam.RVs[8:12],thatCam.RVs[8:12]))
JD_avg[1] = np.average(thisStar.exposures.JDs[8:12])
BCV_avg[1] = np.average(thisStar.exposures.rel_baryVels[8:12])
rv_avg[1] = np.average(arr)
rv_std[1] = np.std(arr)

arr = np.hstack((thisCam.RVs[12:17],thatCam.RVs[12:17]))
JD_avg[2] = np.average(thisStar.exposures.JDs[12:17])
JD_avg[2] = np.average(thisStar.exposures.JDs[12])
BCV_avg[2] = np.average(thisStar.exposures.rel_baryVels[12:17])
rv_avg[2] = np.average(arr)
rv_std[2] = np.std(arr)

arr = np.hstack((thisCam.RVs[17:20],thatCam.RVs[17:20]))
JD_avg[3] = np.average(thisStar.exposures.JDs[17:20])
JD_avg[3] = np.average(thisStar.exposures.JDs[19])
BCV_avg[3] = np.average(thisStar.exposures.rel_baryVels[17:20])
rv_avg[3] = np.average(arr)
rv_std[3] = np.std(arr)

arr = np.hstack((thisCam.RVs[20:24],thatCam.RVs[20:24]))
JD_avg[4] = np.average(thisStar.exposures.JDs[20:24])
BCV_avg[4] = np.average(thisStar.exposures.rel_baryVels[20:24])
rv_avg[4] = np.average(arr)
rv_std[4] = np.std(arr)

# <codecell>

#all RVs from all camera for a single target - bary corrected. 
i=0

x_sine  = np.linspace(np.min(thisStar.exposures.JDs), np.max(thisStar.exposures.JDs))
# for i in np.arange(1,3,0.5):
period = 4.820200
peri_arg = 269.3
# phase = 19299.110-.5+period/360*peri_arg-0.35
phase = -0.056
y_sine = 26100*np.sin(x_sine*2*np.pi/period+phase)
y_sine -= y_sine[0] 
# y_sine -=  13639.225889220206+5000
plt.plot(x_sine, y_sine, c='k')

# for i, thisCam in enumerate(thisStar.exposures.cameras[:2]):
# i=3   camera index
#     thisCam = thisStar.exposures.cameras[i]

#just RV for single camera
#     plt.scatter(thisStar.exposures.JDs, (thisCam.RVs), c=colors[i], label = cameras[i])

#     plt.scatter(thisStar.exposures.JDs, (thisCam.RVs + thisStar.exposures.rel_baryVels), c=colors[i], label = cameras[i])

# average rvs and error bars
plt.errorbar(JD_avg, (rv_avg+BCV_avg), yerr = rv_std, c='b', label = cameras[i], fmt='.')




# plt.scatter(thisStar.exposures.JDs[[0,8,11,17,20]], (rv_avg + thisStar.exposures.rel_baryVels[[0,8,11,17,20]]), c='r', label = cameras[i])


# plt.scatter(thisStar.exposures.JDs[[0,1,4,7,-1]], RV_iraf + thisStar.exposures.rel_baryVels[[0,1,4,7,-1]], c='k', marker = '+')
# plt.scatter(thisStar.exposures.JDs[[0,1,4,7,-1]], RV_iraf, c='k', marker = '+')
# plt.plot(thisStar.exposures.JDs, - thisStar.exposures.rel_baryVels)
plt.title('rhoTuc')
plt.ylabel('RV [m/s]')
plt.xlabel('MJD')
# plt.legend(loc = 0)
plt.show()

# <codecell>

#all red fluxes for the active camera
off = 0
for x,y,label in zip(thisCam.wavelengths, thisCam.red_fluxes, thisCam.fileNames):
    plt.plot(x,y/np.median(y)off, label= label)
    off+=1
plt.title(thisStar.name)
plt.legend(loc = 0)
plt.show()

# <codecell>

#creates images for all cameras from all red fluxes

for cam, thisCam in enumerate(thisStar.exposures.cameras):
    for x,y,label,i in zip(thisCam.wavelengths, thisCam.red_fluxes, thisCam.fileNames, range(thisCam.wavelengths.shape[0])):
        __,y1 = RVT.clean_flux(x,y,thisCam)
        plt.plot(x,y1+i, label= label, c='k')
    plt.title(thisStar.name)
    plt.yticks = thisCam.fileNames 
    plt.savefig('plots/HD1581_'+str(cam), dpi = 1000 )
    
    plt.close()
#     plt.show()

# <codecell>

cd /Users/Carlos/Documents/HERMES/reductions/HD1581_6.0/single_exposures/cam1/

# <codecell>

rv_avg = np.ones(5)*np.nan
rv_avg[0] = np.average(thisCam.RVs[0])
rv_avg[1] = np.average(thisCam.RVs[[1,2,3]])
rv_avg[2] = np.average(thisCam.RVs[[4,5,6]])
rv_avg[3] = np.average(thisCam.RVs[[7,8,9,10,11]])
rv_avg[4] = np.average(thisCam.RVs[[12,13,14]])

# <codecell>

RV_iraf

# <codecell>

#all RVs from all cameras for a single target
for i, thisCam in enumerate(thisStar.exposures.cameras):
    plt.scatter(thisStar.exposures.JDs, thisCam.RVs, c=colors[i], label = cameras[i])
    plt.plot(thisStar.exposures.JDs, - thisStar.exposures.rel_baryVels)

plt.title('HD1581 - RVs')
plt.ylabel('RV [m/s]')
plt.xlabel('JD')
plt.legend(loc = 0)
plt.show()

# <codecell>

#all SNRs from all cameras for a single target
for i, thisCam in enumerate(thisStar.exposures.cameras):
    plt.scatter(thisStar.exposures.JDs, thisCam.SNRs, c=colors[i], label = cameras[i])
plt.title('HD1581 - Signal-to-Noise')
plt.ylabel('SNR')
plt.xlabel('JD')
plt.legend(loc = 0)
plt.show()

# <codecell>

#all Q from all cameras for a single target
for i, thisCam in enumerate(thisStar.exposures.cameras):
    plt.scatter(thisStar.exposures.JDs, thisCam.Qs, c=colors[i], label = cameras[i], marker = '+', s=50)
plt.title('HD1581 - Q-factor')
plt.ylabel('Q')
plt.xlabel('JD')
plt.legend(loc = 0)
plt.show()

# <codecell>

#all dRV from all cameras for a single target
for i, thisCam in enumerate(thisStar.exposures.cameras):
    plt.scatter(thisStar.exposures.JDs, thisCam.sigmas, c=colors[i], label = cameras[i], marker = 'x', s=50)
plt.title('HD1581 - Photon limited dRV')
plt.ylabel('dRV [m/s]')
plt.xlabel('JD')
plt.legend(loc = 0)
plt.show()

# <headingcell level=1>

# Random stuff....

# <codecell>

myTarget=36

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

1/10e-6/3600

# <codecell>


