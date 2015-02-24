# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#general imports
import glob
import os
import numpy as np
from scipy import signal, interpolate, optimize, constants
import pylab as plt
import pickle
import pyfits as pf

#my imports
import create_obj as cr_obj
reload(cr_obj)
import RVTools as RVT
reload(RVT)

# <codecell>

#single exposure data reduction

# <codecell>

pwd

# <codecell>

cd /Users/Carlos/Documents/HERMES/reductions/HD1581_6.0/uncombined/

# <codecell>

#uses create_obj.py to create star with basic reduced data

thisStar = cr_obj.star('Giant01')
thisStar.exposures = cr_obj.exposures()
thisStar.exposures.load_exposures(thisStar.name)
thisStar.exposures.calculate_baryVels(thisStar)
thisStar.name = 'HD1581'
file_pi = open('HD1581.obj', 'w') 
pickle.dump(thisStar, file_pi) 
file_pi.close()

# <codecell>

#takes each <star_name>.obj and calculates RVs, etc. Saves red_<star_name>.obj

xDef = 10 #resampling points per pixel
CCMaskWidth = 5 #half width around the peak of the cc curve to be gaussian fitted

filename = 'HD1581.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

RVT.find_max_wl_range(thisStar)
RVT.RVs_CC_t0(thisStar)
file_pi = open('red_HD1581.obj', 'w') 
pickle.dump(thisStar, file_pi) 
file_pi.close()

# <codecell>

#takes each <star_name>.obj and calculates RVs, etc. Saves red_<star_name>.obj
reload(RVT)
xDef = 10 #resampling points per pixel
CCMaskWidth = 5 #half width around the peak of the cc curve to be gaussian fitted

filename = 'HD1581.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

RVT.find_max_wl_range(thisStar)
RVT.RVs_CC_t0(thisStar)
file_pi = open('red_HD1581.obj', 'w') 
pickle.dump(thisStar, file_pi) 
file_pi.close()

# <headingcell level=3>

# Collect reduced data into single object

# <codecell>

#uses create_obj.py to create star with basic reduced data
thisStar = cr_obj.star('Giant01')
thisStar.exposures = cr_obj.exposures()
thisStar.exposures.load_exposures(thisStar.name)
thisStar.exposures.calculate_baryVels(thisStar)
thisStar.name = 'HD1581'
file_pi = open('HD1581.obj', 'w') 
pickle.dump(thisStar, file_pi) 
file_pi.close()

# <headingcell level=3>

# Take initial data from HD1581.obj and create red_HD1581.obj

# <codecell>

#takes each <star_name>.obj and calculates RVs, etc. Saves red_<star_name>.obj
reload(RVT)
xDef = 10 #resampling points per pixel
CCMaskWidth = 5 #half width around the peak of the cc curve to be gaussian fitted

filename = 'HD1581.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

RVT.find_max_wl_range(thisStar)
RVT.RVs_CC_t0(thisStar)
file_pi = open('red_HD1581.obj', 'w') 
pickle.dump(thisStar, file_pi) 
file_pi.close()

# <headingcell level=3>

# IRAF import from data reduced by gayandhi

# <codecell>

file0_wl = '/Users/Carlos/Documents/HERMES/reductions/HD1581_iraf/norm/hd1581_0.fits'
file1_wl = '/Users/Carlos/Documents/HERMES/reductions/HD1581_iraf/norm/hd1581_1.fits'
file2_wl = '/Users/Carlos/Documents/HERMES/reductions/HD1581_iraf/norm/hd1581_2.fits'
file3_wl = '/Users/Carlos/Documents/HERMES/reductions/HD1581_iraf/norm/hd1581_3.fits'
file4_wl = '/Users/Carlos/Documents/HERMES/reductions/HD1581_iraf/norm/hd1581_4.fits'

file0 = '/Users/Carlos/Documents/HERMES/reductions/HD1581_iraf/px/0_20augt.fits'
file1 = '/Users/Carlos/Documents/HERMES/reductions/HD1581_iraf/px/1_21augt.fits'
file2 = '/Users/Carlos/Documents/HERMES/reductions/HD1581_iraf/px/2_22augt.fits'
file3 = '/Users/Carlos/Documents/HERMES/reductions/HD1581_iraf/px/3_24augt.fits'
file4 = '/Users/Carlos/Documents/HERMES/reductions/HD1581_iraf/px/4_25augt.fits'

a_wl = pf.open(file0_wl)
b_wl = pf.open(file1_wl)
c_wl = pf.open(file2_wl)
d_wl = pf.open(file3_wl)
e_wl = pf.open(file4_wl)
a_flux_wl = np.array(a_wl[0].data.astype(float))
b_flux_wl = np.array(b_wl[0].data.astype(float))
c_flux_wl = np.array(c_wl[0].data.astype(float))
d_flux_wl = np.array(d_wl[0].data.astype(float))
e_flux_wl = np.array(e_wl[0].data.astype(float))

a = pf.open(file0)
b = pf.open(file1)
c = pf.open(file2)
d = pf.open(file3)
e = pf.open(file4)
a_flux = np.array(a[0].data.astype(float))
b_flux = np.array(b[0].data.astype(float))
c_flux = np.array(c[0].data.astype(float))
d_flux = np.array(d[0].data.astype(float))
e_flux = np.array(e[0].data.astype(float))
# a_lambdas = extract_HERMES_wavelength(file1)
# b_lambdas = extract_HERMES_wavelength(file2)

RV_iraf = np.ones(5) * np.nan

a_flux_wl = a_flux_wl-1
b_flux_wl = b_flux_wl-1
c_flux_wl = c_flux_wl-1
d_flux_wl = d_flux_wl-1
e_flux_wl = e_flux_wl-1


a_flux = a_flux-1
b_flux = b_flux-1
c_flux = c_flux-1
d_flux = d_flux-1
e_flux = e_flux-1

# <codecell>

e_flux_wl

# <codecell>

a = pf.open('/Users/Carlos/Documents/HERMES/reductions/HD1581_6.0/0_20aug/1/20aug10053red.fits') 

CRVAL1 = a[0].header['CRVAL1'] # / Co-ordinate value of axis 1                    
CDELT1 = a[0].header['CDELT1'] #  / Co-ordinate increment along axis 1             
CRPIX1 = a[0].header['CRPIX1'] #  / Reference pixel along axis 1                   

CRVAL1, CDELT1, CRPIX1

ms_px = CDELT1/CRVAL1*constants.c

# <codecell>

# # a_flux_c = a_flux[:-145]
# # b_flux_c = b_flux[:-145]
# __,a_flux_c = fit_continuum(np.arange(a_flux.shape[0]),a_flux)
# __,b_flux_c = fit_continuum(np.arange(b_flux.shape[0]),b_flux)
# a_flux_c = a_flux_c-1
# b_flux_c = b_flux_c-1

# <codecell>

# plt.plot(a_flux, label=os.path.basename(file0) )
# plt.plot(b_flux, label=os.path.basename(file1) )
# plt.plot(c_flux, label=os.path.basename(file2) )
# plt.plot(d_flux, label=os.path.basename(file3) )
# plt.plot(e_flux, label=os.path.basename(file4) )
plt.plot(a_flux_wl, label=os.path.basename(file0) )
plt.plot(b_flux_wl, label=os.path.basename(file1) )
plt.plot(c_flux_wl, label=os.path.basename(file2) )
# plt.plot(d_flux_wl, label=os.path.basename(file3) )
plt.plot(e_flux_wl, label=os.path.basename(file4) )
plt.legend(loc=0)
plt.show()

# <codecell>

cc = signal.convolve(a_flux, a_flux[::-1], mode='same')
# cc = signal.convolve(a_flux_wl, a_flux_wl[::-1], mode='same')

corrMax = np.where(cc==max(cc))[0][0]
idx = (np.abs(cc - max(cc)*.9)).argmin()
h_range = 2

p_guess = [corrMax,h_range]
x_mask = np.arange(corrMax-h_range, corrMax+h_range+1)
p = RVT.fit_gaussian(p_guess, cc[x_mask], np.arange(len(cc))[x_mask])[0]

print 't0 wrt t0'
print p
print (cc.shape[0])/2.
print p[0]-(cc.shape[0])/2.
RV_iraf[0] = (p[0]-(cc.shape[0])/2.)*ms_px
print RV_iraf[0], 'm/s'

# <codecell>

cc = signal.convolve(a_flux, b_flux[::-1], mode='same' )
# cc = signal.convolve(a_flux_wl, b_flux_wl[::-1], mode='same')

corrMax = np.where(cc==max(cc))[0][0]
idx = (np.abs(cc - max(cc)*.9)).argmin()
h_range = int(abs(idx - corrMax))
h_range = 2

p_guess = [corrMax,h_range]
x_mask = np.arange(corrMax-h_range, corrMax+h_range+1)
p = RVT.fit_gaussian(p_guess, cc[x_mask], np.arange(len(cc))[x_mask])[0]

print 't0 wrt t1'
print p
print (cc.shape[0])/2.
print p[0]-(cc.shape[0])/2.
RV_iraf[1] = (p[0]-(cc.shape[0])/2.)*ms_px
print RV_iraf[1], 'm/s'

# <codecell>

cc = signal.convolve(a_flux, c_flux[::-1], mode='same' )
# cc = signal.convolve(a_flux_wl, c_flux_wl[::-1], mode='same')

corrMax = np.where(cc==max(cc))[0][0]
idx = (np.abs(cc - max(cc)*.9)).argmin()
h_range = 2

h_range = int(abs(idx - corrMax))
p_guess = [corrMax,h_range]
x_mask = np.arange(corrMax-h_range, corrMax+h_range+1)
p = RVT.fit_gaussian(p_guess, cc[x_mask], np.arange(len(cc))[x_mask])[0]

print 't0 wrt t2'
print p
print (cc.shape[0])/2.
print p[0]-(cc.shape[0])/2.
RV_iraf[2] = (p[0]-(cc.shape[0])/2.)*ms_px
print RV_iraf[2], 'm/s'

# <codecell>

cc = signal.convolve(a_flux, d_flux[::-1], mode='same' )
# cc = signal.convolve(a_flux_wl, d_flux_wl[::-1], mode='same')

corrMax = np.where(cc==max(cc))[0][0]
idx = (np.abs(cc - max(cc)*.9)).argmin()
h_range = 2

h_range = int(abs(idx - corrMax))
p_guess = [corrMax,h_range]
x_mask = np.arange(corrMax-h_range, corrMax+h_range+1)
p = RVT.fit_gaussian(p_guess, cc[x_mask], np.arange(len(cc))[x_mask])[0]

print 't0 wrt t2'
print p
print (cc.shape[0])/2.
print p[0]-(cc.shape[0])/2.
RV_iraf[3] = (p[0]-(cc.shape[0])/2.)*ms_px
print RV_iraf[3], 'm/s'

# <codecell>

cc = signal.convolve(a_flux, e_flux[::-1], mode='same' )
# cc = signal.convolve(a_flux_wl, e_flux_wl[::-1], mode='same')

corrMax = np.where(cc==max(cc))[0][0]
idx = (np.abs(cc - max(cc)*.9)).argmin()
h_range = int(abs(idx - corrMax))
h_range = 2

p_guess = [corrMax,h_range]
x_mask = np.arange(corrMax-h_range, corrMax+h_range+1)
p = RVT.fit_gaussian(p_guess, cc[x_mask], np.arange(len(cc))[x_mask])[0]

print 't0 wrt t4'
print p
print (cc.shape[0])/2.
print p[0]-(cc.shape[0])/2.
RV_iraf[4] = (p[0]-(cc.shape[0])/2.)*ms_px
print RV_iraf[4], 'm/s'

# <headingcell level=3>

# Plots, sanity checks, results

# <codecell>

colors = ['b','g','r','cyan']
cameras = ['Blue', 'Green', 'Red', 'IR']

# <codecell>

filename = 'red_HD1581.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)
thisCam=thisStar.exposures.cameras[0]

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

#all red fluxes for the active camera
for x,y,label in zip(thisCam.wavelengths, thisCam.red_fluxes, thisCam.fileNames):
    plt.plot(x,y, label= label)
plt.title(thisStar.name)
plt.legend(loc = 0)
plt.show()

# <codecell>

cd /Users/Carlos/Documents/HERMES/reductions/HD1581_6.0/single_exposures/cam1/

# <codecell>

thisCam=thisStar.exposures.cameras[0]

rv_avg = np.ones(5)*np.nan
rv_std = np.ones(5)*np.nan
rv_avg[0] = np.average(thisCam.RVs[0])
rv_std[0] = np.std(thisCam.RVs[0])
rv_avg[1] = np.average(thisCam.RVs[[1,2,3]])
rv_std[1] = np.std(thisCam.RVs[[1,2,3]])
rv_avg[2] = np.average(thisCam.RVs[[4,5,6]])
rv_std[2] = np.std(thisCam.RVs[[4,5,6]])
rv_avg[3] = np.average(thisCam.RVs[[7,8,9,10,11]])
rv_std[3] = np.std(thisCam.RVs[[7,8,9,10,11]])
rv_avg[4] = np.average(thisCam.RVs[[12,13,14]])
rv_std[4] = np.std(thisCam.RVs[[12,13,14]])

# <codecell>

print 'Green RV std for ALL 5 POINTS:',  np.std(rv_avg)

# <codecell>

print 'Red RV std for ALL 5 POINTS:',  np.std(rv_avg)

# <codecell>

#all RVs from all camera for a single target - bary corrected. 

# for i, thisCam in enumerate(thisStar.exposures.cameras):
i=0
#     thisCam = thisStar.exposures.cameras[i]
# plt.scatter(thisStar.exposures.JDs, (thisCam.RVs + thisStar.exposures.rel_baryVels), c=colors[1], label = cameras[i])

# plt.scatter(thisStar.exposures.JDs[[0,1,4,7,-1]], (rv_avg + thisStar.exposures.rel_baryVels[[0,1,4,7,-1]]), c=colors[i], label = cameras[i])

# average rvs and error bars
plt.errorbar(thisStar.exposures.JDs[[0,1,4,7,-1]], (rv_avg + thisStar.exposures.rel_baryVels[[0,1,4,7,-1]]), yerr = rv_std, c=colors[0], label = cameras[i], fmt='.')
plt.plot(thisStar.exposures.JDs[[0,-1]], [0,0], 'k--')
# plt.scatter(thisStar.exposures.JDs, (thisCam.RVs ), c=colors[i], label = cameras[i])
# plt.scatter(thisStar.exposures.JDs[[0,1,4,7,-1]], RV_iraf + thisStar.exposures.rel_baryVels[[0,1,4,7,-1]], c='k', marker = '+')
# plt.scatter(thisStar.exposures.JDs[[0,1,4,7,-1]], RV_iraf, c='k', marker = '+')
# plt.plot(thisStar.exposures.JDs, - thisStar.exposures.rel_baryVels)
plt.title('HD1581')
plt.ylabel('RV [m/s]')
plt.xlabel('MJD')
# plt.legend(loc = 0)
plt.show()

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


