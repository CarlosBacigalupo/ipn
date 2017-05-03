
# coding: utf-8

# In[3]:

import pylab as plt
import numpy as np
import glob 
import os
import subprocess 
import toolbox
import scipy.optimize as opt
import pyfits as pf

# import HERMES
# reload(HERMES)


# In[6]:

baseDir = '/Users/Carlos/Documents/HERMES/reductions/PSF/LR'
os.chdir(baseDir)


# In[13]:

a = HERMES.PSF()
a.sexParamFile = 'HERMES.sex'
a.sex_path = '/usr/local/bin/'
a.outputFileName = 'out.txt'
# a.nFibres = 10

# a.scienceFile = '07feb' + str(a.camera) + '0022.fits'
# a.biasFile = 'BIAScombined2.fits'
a.profile = 'gaussian'
# a.flatFile = '07feb' + str(a.camera) + '0020.fits'
# a.tramLinefile = '07feb' + str(a.camera) + '0020tlm.fits'

a.camera=4
# a.arcFile = '07feb' + str(a.camera) + '0021.fits'
a.arcFile = '05aug' + str(a.camera) + '0016.fits'


# In[14]:

#analise all images in current folder (create PSF text file)
a.read_full_image_spectral('gaussian')


# In[4]:

#creates PSF size  plots from sigma size
# files = glob.glob('point_maps/*.txt')

# for dataFile in files:
#     data = np.loadtxt(dataFile, skiprows=1, delimiter = ',' , usecols=[0,1,2,3,4,5])
#     print dataFile
            
#     plt.scatter(data[:,0],data[:,1], s=data[:,3]/np.max(data[:,3])*10, edgecolors='none')
#     plt.savefig(dataFile[:-19],dpi=700.)
# #     plt.show()
#     plt.close()


# In[99]:

#crates bins for data analisys per file
#X AXIS BUNDLE NUMBER
#Y AXIS PSF SIZE
#SERIES CCD X-PIXEL
# one per file
os.chdir('/Users/Carlos/Documents/HERMES/reductions/PSF/spectral')
a = pf.open('10nov10044tlm.fits')
bundleBins = a[0].data[range(0,399,10),0]-5
xBins = range(0,a[0].shape[1]-1,100)
# bundleBins = bundleBins[::5]
a= None

#reads PSF output data and creates final arrays
os.chdir('/Users/Carlos/Documents/HERMES/reductions/PSF/spectral/cam4')
files = glob.glob('point_maps/*.txt')
for i in range(len(files[:])):
    sigmas = np.zeros((len(bundleBins),len(xBins),len(files)))
    print files[i]
    thisData = np.loadtxt(files[i], skiprows=1, delimiter = ',' , usecols=[0,1,2,3,4,5])
    bundleIdx = np.digitize(thisData[:,1], bundleBins)
    xIdx = np.digitize(thisData[:,0], xBins)
    for r in range(1,41):
        for c in range(1,41):
            subset = thisData[((bundleIdx==r) & (xIdx==c))]
            if subset.shape[0]>1:
                sigmas[r-1,c-1,i]=np.mean(subset[:,3])

    FWHM = 2 * np.sqrt(2*np.log(2)) * sigmas
    
    #clean outliers and find error bars
#     sigmas[sigmas<0.5]=np.nan
#     sigmas[sigmas>3]=np.nan
#     FWHM[FWHM<1.5]=np.nan
#     FWHM[FWHM>6]=np.nan
    errorsSig = np.std(sigmas,axis=2)
    meanSig = np.mean(sigmas, axis=2)
    errorsFWHM = np.std(FWHM,axis=2)
    meanFWHM = np.mean(FWHM, axis=2)
    meanFWHM = FWHM[:,:,i]
    FWHM[FWHM<1.5]=np.nan
    
    levels = xBins#np.arange(-0.5, 2., 1.)
    CS = plt.contour(meanFWHM, levels,
                 origin='lower',
                 cmap=plt.cm.rainbow, 
                 linewidths=5)#,

    colors = iter(plt.cm.rainbow(np.linspace(0, 1, sigmas.shape[0])))
    #FWHM + errors
    for j in range(sigmas.shape[0]):
#         plt.errorbar(range(40),meanFWHM[:,i],yerr=errorsFWHM[:,i], fmt='.', label = 'Range(x) = ' + str(xBins[i]))   
        plt.scatter(range(len(bundleBins)),meanFWHM[:,j],color=next(colors), label = xBins[j])

    CB = plt.colorbar(CS, shrink=1, extend='both')
    
#     plt.gray()  Now change the colormap for the contour lines and colorbar
    #         plt.flag()
    
    # We can still add a colorbar for the image, too.
#     CBI = plt.colorbar(im, orientation='vertical', shrink=1)
    
    # This makes the original colorbar look a bit out of place,
    # so let's improve its position.
    
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB.ax.get_position().bounds
    CB.ax.set_position([ll, b+0.1*h, ww, h*0.8])

    plt.title('Spectral PSF - ' + os.path.basename(files[i])[:-19])
    plt.xlabel('Fibre Bundle')
    plt.ylabel('FWHM (Pixel)')
#     plt.legend(loc='best', ncol=2,  bbox_to_anchor=(1.2, 1))
    plt.axis([-1, 41, 1.5, 8])
    plt.savefig(os.path.basename(files[i])[:-19]+'_bundle')
    plt.close()
#     plt.show()


# In[113]:

# plt.errorbar(range(len(bundleBins)),meanFWHM[j,:],yerr=errorsFWHM[j,:], fmt='.')
# sigmas.shape
os.chdir('/Users/Carlos/Documents/HERMES/reductions/PSF/spectral')
a = pf.open('10nov10044tlm.fits')
a[0].shape
len(range(0,a[0].shape[1]-1,100))


# In[51]:

#crates bins for data analisys
#X AXIS x pixel
#Y AXIS PSF SIZE
#SERIES Bundle number
os.chdir('/Users/Carlos/Documents/HERMES/reductions/PSF/spectral')
a = pf.open('10nov10044tlm.fits')
bundleBins = a[0].data[range(0,399,10),0]-5
xBins = range(0,a[0].shape[1]-1,100)
# bundleBins = bundleBins[::5]
a= None

#reads PSF output data and creates final arrays
os.chdir('/Users/Carlos/Documents/HERMES/reductions/PSF/spectral/cam4')
files = glob.glob('point_maps/*.txt')
sigmas = np.zeros((len(bundleBins),len(xBins),len(files)))
for i in range(len(files[:])):
    print files[i]
    thisData = np.loadtxt(files[i], skiprows=1, delimiter = ',' , usecols=[0,1,2,3,4,5])
    bundleIdx = np.digitize(thisData[:,1], bundleBins)
    xIdx = np.digitize(thisData[:,0], xBins)
    for r in range(1,41):
        for c in range(1,41):
            subset = thisData[((bundleIdx==r) & (xIdx==c))]
            if subset.shape[0]>1:
                sigmas[r-1,c-1,i]=np.mean(subset[:,3])

FWHM = 2 * np.sqrt(2*np.log(2)) * sigmas


# In[52]:


#clean outliers and find error bars
#     sigmas[sigmas<0.5]=np.nan
#     sigmas[sigmas>3]=np.nan
FWHM[FWHM<1.5]=np.nan
FWHM[FWHM>7]=np.nan

# errorsSig = np.std(sigmas,axis=2)
# meanSig = np.mean(sigmas, axis=2)
# meanFWHM = np.mean(FWHM[FWHM!=0], axis=2)
FWHM[FWHM==0]=np.nan
mdat = np.ma.masked_array(FWHM,np.isnan(FWHM))
errorsFWHM = np.std(mdat,axis=2).filled(np.nan)
meanFWHM = np.mean(mdat,axis=2).filled(np.nan)

levels = range(0,len(bundleBins)+1)#np.arange(-0.5, 2., 1.)
# CS = plt.contour(meanFWHM, levels,
#              origin='lower',
#              cmap=plt.cm.rainbow, 
#              linewidths=5)#,

# colors = iter(plt.cm.rainbow(np.linspace(0, 1, sigmas.shape[0])))
#FWHM + errors
for j in range(sigmas.shape[0]):

    plt.errorbar(xBins,meanFWHM[j,:],yerr=errorsFWHM[j,:], fmt = '.') #
# print xBins,meanFWHM[j,:].shape
#     plt.scatter(range(len(bundleBins)),meanFWHM[j,:],color=next(colors), label = bundleBins[j])

CB = plt.colorbar(CS, shrink=1, extend='both')

#     plt.gray()  Now change the colormap for the contour lines and colorbar
#         plt.flag()

# We can still add a colorbar for the image, too.
#     CBI = plt.colorbar(im, orientation='vertical', shrink=1)

# This makes the original colorbar look a bit out of place,
# so let's improve its position.

# l,b,w,h = plt.gca().get_position().bounds
# ll,bb,ww,hh = CB.ax.get_position().bounds
# CB.ax.set_position([ll, b+0.1*h, ww, h*0.8])

plt.title('Spectral PSF - All HR frames - IR')
plt.xlabel('x-Pixel')
plt.ylabel('FWHM (Pixel)')
#     plt.legend(loc='best', ncol=2,  bbox_to_anchor=(1.2, 1))
# plt.axis([-1, 41, 0, 8])
plt.savefig('All_HR_mean_cam4')
# plt.close()
plt.show()


# In[20]:

# FWHM[FWHM==0]=np.nan
for r in range(0,22):
#     for c in range(0,40):
    print np.max(sigmas[:,:,r])
# ow()


# In[23]:

# os.path.basename(f)[:-19]


# In[27]:

#fractional errors
colors = iter(plt.cm.rainbow(np.linspace(0, 1, sigmas.shape[0])))
# for y in ys:
#     plt.scatter(x, y, color=next(colors))
for i in range(sigmas.shape[0]):
    plt.scatter(xBins,errorsFWHM[i,:],color=next(colors), label = bundleBins[i])
plt.legend()
plt.title('Spectral PSF - Fractional Errors')
plt.xlabel('Pixel')
plt.ylabel('FWHM (Pixel)')    
plt.show()


# In[53]:

bundleBins,(sigmas.shape[0])
             


# In[38]:

# plt.imshow(sigmas[:,:,10], origin='lower')
# plt.show()


# In[40]:

plt.imshow(mean, origin='lower')
plt.show()


# ###### Spatial PSF read

# In[ ]:

baseDir = '/Users/Carlos/Documents/HERMES/reductions/PSF/spatial/'
os.chdir(baseDir)


# In[ ]:

nColumns = 10
Width = 4000
columns = range(0, Width,Width/nColumns)


# In[ ]:

# a.open_files()


# In[ ]:

a.read_full_image_spatial('gaussian',[10])


# In[ ]:

#bias subtract using the last 40 columns(overscan region)
# a.scienceIm_b = a.bias_subtract_from_overscan(a.scienceIm, range(-45,-5))


# In[ ]:

# plt.plot(np.sum(a.flatIm,0))
plt.plot(np.sum(a.flatIm,1))
plt.plot(a.flatIm[:,1000])
plt.show()


# In[ ]:

a.profile = 'voigt'
sigmaV = np.zeros(a.imWidth)
gammaV = np.zeros(a.imWidth)


# In[12]:

type(a)


# In[ ]:

a.fit_10f(a.scienceIm_b[:,10])


# In[ ]:

for i in range(0,a.imWidth,100):
    sigmaV[i], gammaV[i] = a.fit_10f(a.scienceIm_b[:,i])
    print i,


# In[ ]:

a.profile = 'gaussian'
sigmaG = np.zeros(a.imWidth+5)
gammaG = np.zeros(a.imWidth+5)


# In[ ]:

a.fit_10f(a.scienceIm_b[:,10])


# In[ ]:

for i in range(0,a.imWidth,100):
    sigmaG[i+5], gammaG[i+5] = a.fit_10f(a.scienceIm_b[:,i])
    print i,


# In[ ]:

plt.plot(sigmaV)
plt.plot(sigmaG)
plt.show()


# In[ ]:

a.profile = 'gaussian'
a.base_dir = '/Users/Carlos/Documents/HERMES/reductions/resolution_gayandhi/'
a.out_dir = a.base_dir + 'output'
a.flatFile = a.base_dir + '10nov10044.fits'
a.nFibres = 400


# In[ ]:

a.open_files()
a.bias_subtract()


# In[ ]:

import HERMES
import numpy as np

psf = HERMES.PSF()

psf.base_dir = '/Users/Carlos/Documents/HERMES/reductions/resolution_gayandhi/'
psf.out_dir = psf.base_dir + 'output/'
# dataFile = open(psf.out_dir + 'spatialG1.txt','r')


# In[ ]:

a = np.loadtxt( 'spectralG1.txt', skiprows=1, delimiter = ',' , usecols=[0,1,2,3,4,5]).transpose()


# In[4]:

import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

X, Y = np.meshgrid(np.unique(a[1]), np.unique(a[0]))
Z = a[3].reshape(X.shape[1],X.shape[0])
Z = Z.transpose()


# In[ ]:

# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')
plt.show()


# In[ ]:

# contour labels can be placed manually by providing list of positions
# (in data coordinate). See ginput_manual_clabel.py for interactive
# placement.
plt.figure()
CS = plt.contour(X, Y, Z)
# manual_locations = [(-1, -1.4), (-0.62, -0.7), (-2, 0.5), (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)]
# plt.clabel(CS, inline=1, fontsize=10, manual=manual_locations)
plt.title('labels at selected locations')
plt.show()


# In[5]:

import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pyfits as pf
import math

#     from scipy.interpolate import interp1d
#     from scipy.interpolate import interp2d
#     import pyfits as pf

cameraName = 'Blue Camera'
    

#         self.base_dir = '/Users/Carlos/Documents/HERMES/reductions/resolution_gayandhi/'
dataFile = 'spectralG1.txt'

data = np.loadtxt(dataFile, skiprows=1, delimiter = ',' , usecols=[0,1,2,3,4,5])

hdulist = pf.open(a.arcFile)
imWidth = hdulist[0].header['NAXIS1']
imHeight = hdulist[0].header['NAXIS2']

#         Lambda = RVS.extract_HERMES_wavelength(referenceFile)

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

#         X, Y = np.meshgrid(a[:,1], a[:,0])
#         X, Y = np.meshgrid(range(imHeight),range(imWidth))
#         Z = a[3].reshape(X.shape[1],X.shape[0])
#         Z = Z.transpose()
Z = np.zeros((imHeight,imWidth))
for r,c,z in zip(data[:,1].astype(int),data[:,0].astype(int), data[:,3]):
    Z[r,c] = z
mean = np.mean(Z)
print mean
Z[Z==0]=mean
#         Z = np.diagflat(a[:,3])
#         Z[Z<0.5] = np.average(Z)
#         Z[Z>4] = np.average(Z)

Z = 2 * np.sqrt(2*math.log(2)) * Z

# Or you can use a colormap to specify the colors; the default
# colormap will be used for the contour lines
# plt.figure()
# im = plt.imshow(Z, interpolation='nearest', origin='lower',
#                 cmap=cm.jet)#, extent=(0,400,1,400))
#         plt.xticks(range(40), Lambda.astype(int)[::len(Lambda)/40], size='small')
#         plt.xticks(range(0,400,100),Lambda.astype(int)[::len(Lambda)/100])
levels = np.arange(-0.5, 2., 1.)
CS = plt.contour(Z, levels,
                 origin='lower',
                 cmap=cm.jet, 
                 linewidths=5)#,
                 #extent=(0,400,1,400))

#Thicken the zero contour.
#         zc = CS.collections[6]
#         plt.setp(zc, linewidth=4)
 
# plt.clabel(CS, levels,  # levels[1::2]  to label every second level
#            inline=0,
#            fmt='%1.2f',
#            fontsize=12)

# make a colorbar for the contour lines
CB = plt.colorbar(CS, shrink=0.8, extend='both')

plt.title('Spectral FWHM - ' + cameraName)
plt.gray()  # Now change the colormap for the contour lines and colorbar
#         plt.flag()

# We can still add a colorbar for the image, too.
# CBI = plt.colorbar(im, orientation='vertical', shrink=1)

# This makes the original colorbar look a bit out of place,
# so let's improve its position.

l,b,w,h = plt.gca().get_position().bounds
ll,bb,ww,hh = CB.ax.get_position().bounds
CB.ax.set_position([ll, b+0.1*h, ww, h*0.8])
plt.xlabel('Pixel')
plt.ylabel('Pixel')

plt.show()


# In[41]:

import scipy


# In[51]:

b = scipy.interpolate.SmoothBivariateSpline(Z[:,0],Z[:,1],Z[:,3])


# In[59]:

np.isnan(b.ev(range(4000),range(4000)))


# In[1]:

import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

# delta = 0.025
# x = np.arange(-3.0, 3.0, delta)
# y = np.arange(-2.0, 2.0, delta)
# X, Y = np.meshgrid(x, y)
# Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
# Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# # difference of Gaussians
# Z = 10.0 * (Z2 - Z1)

# # Create a simple contour plot with labels using default colors.  The
# # inline argument to clabel will control whether the labels are draw
# # over the line segments of the contour, removing the lines beneath
# # the label
# plt.figure()
# CS = plt.contour(X, Y, Z)
# plt.clabel(CS, inline=1, fontsize=10)
# plt.title('Simplest default with labels')


# # contour labels can be placed manually by providing list of positions
# # (in data coordinate). See ginput_manual_clabel.py for interactive
# # placement.
# plt.figure()
# CS = plt.contour(X, Y, Z)
# manual_locations = [(-1, -1.4), (-0.62, -0.7), (-2, 0.5), (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)]
# plt.clabel(CS, inline=10, fontsize=10, manual=manual_locations)
# plt.title('labels at selected locations')


# # You can force all the contours to be the same color.
# plt.figure()
# CS = plt.contour(X, Y, Z, 6,
#                  colors='k', # negative contours will be dashed by default
#                  )
# plt.clabel(CS, fontsize=9, inline=1)
# plt.title('Single color - negative contours dashed')

# # You can set negative contours to be solid instead of dashed:
# matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
# plt.figure()
# CS = plt.contour(X, Y, Z, 6,
#                  colors='k', # negative contours will be dashed by default
#                  )
# plt.clabel(CS, fontsize=9, inline=1)
# plt.title('Single color - negative contours solid')


# # And you can manually specify the colors of the contour
# plt.figure()
# CS = plt.contour(X, Y, Z, 6,
#                  linewidths=np.arange(.5, 4, .5),
#                  colors=('r', 'green', 'blue', (1,1,0), '#afeeee', '0.5')
#                  )
# plt.clabel(CS, fontsize=9, inline=1)
# plt.title('Crazy lines')


# Or you can use a colormap to specify the colors; the default
# colormap will be used for the contour lines
plt.figure()
# im = plt.imshow(Z, interpolation='nearest', origin='lower',
#                 cmap=cm.jet, extent=(0,400,0,400))
levels = np.arange(0, 2.2, 0.1)
CS = plt.contour(Z, levels,
                 origin='lower',
                 cmap=cm.jet, 
                 linewidths=2,
                 extent=(0,400,0,400))

#Thicken the zero contour.
zc = CS.collections[6]
plt.setp(zc, linewidth=4)

plt.clabel(CS, levels[1::2],  # label every second level
           inline=1,
           fmt='%1.1f',
           fontsize=14)

# make a colorbar for the contour lines
CB = plt.colorbar(CS, shrink=0.8, extend='both')

plt.title('Lines with colorbar')
#plt.hot()  # Now change the colormap for the contour lines and colorbar
plt.flag()

# We can still add a colorbar for the image, too.
CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)

# This makes the original colorbar look a bit out of place,
# so let's improve its position.

l,b,w,h = plt.gca().get_position().bounds
ll,bb,ww,hh = CB.ax.get_position().bounds
CB.ax.set_position([ll, b+0.1*h, ww, h*0.8])


plt.show()


# In[ ]:

a = np.zeros((40,40))*np.nan


# In[ ]:

a[1,1]=1


# In[ ]:

a[20,30]=1


# In[ ]:

Z


# In[ ]:

plt.imshow(a, interpolation='quadric')
plt.show()
plt.imshow(a, interpolation='catrom')
plt.show()
plt.imshow(a, interpolation='gaussian')
plt.show()
plt.imshow(a, interpolation='bessel')
plt.show()
plt.imshow(a, interpolation='mitchell')
plt.show()
plt.imshow(a, interpolation='sinc')
plt.show()
plt.imshow(a, interpolation='lanczos')
plt.show()
plt.imshow(a, interpolation='kaiser')
plt.show()


# In[ ]:



