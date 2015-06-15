# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pylab as plt
import sys
from matplotlib import gridspec

# <codecell>

def onpick1(event):
    if isinstance(event.artist, plt.Line2D):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        x_point = np.take(xdata, ind)[0]
        y_point = np.take(ydata, ind)[0]
        print' Selected:', data[ind[0]]
        print
        ax1.plot(FIBRE[ind[0]], SNR[ind[0]], 'o', picker=5, c='r')
        ax2.plot(FIBRE[ind[0]], RV[ind[0]], 'o', picker=5, c='r')
        ax4.plot(RV[ind[0]], SNR[ind[0]], 'o', picker=5, c='r')
        ax5.plot(RA[ind[0]], DEC[ind[0]], 'o', picker=5, c='r')
        plt.draw()

        

# <codecell>

if len(sys.argv)>2:
    cam = int(sys.argv[1])
    epoch = int(sys.argv[2])

    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
    baryVels=np.load('npy/baryVels.npy')
    SNRs = np.load('npy/SNRs.npy')
    #     JDs=np.load('npy/JDs.npy')
    #     sigmas=np.load('npy/sigmas.npy')
    
    RVClip = 1e9
    if len(sys.argv)>3:
        RVClip = sys.argv[3]
        
    FIBRE = data[:,2].astype(float) #fibres
    RA = data[:,3].astype(float) #RA
    DEC = data[:,4].astype(float) #Dec
    RV = RVs[:,epoch,cam] #RV
    SNR = SNRs[:,epoch,cam] #SNR
    # baryVel = baryVels[:,epoch,cam] #baryVels
    maxSNRs = np.nanmax(SNRs,axis=(0,1))
    
    RVfilter = np.abs(RV)>int(RVClip)
    RV[RVfilter]=np.nan

    fig = plt.figure()
    
    gs = gridspec.GridSpec(3, 4)

#     ax1 = fig.add_subplot(321) #SNRs
    ax1 = fig.add_subplot(gs[0,0:2]) #SNRs
    ax1.plot(FIBRE, SNR, 'o', label = 'SNR', color = 'b', picker=5)
    plt.xlabel('Fibre #')
    plt.ylabel('SNR')
#     plt.legend()


#     ax2= fig.add_subplot(323) #RVs
    ax2= fig.add_subplot(gs[1,0:2]) #RVs
    ax2.plot(FIBRE, RV, 'o', label = 'RV', color = 'b', picker=5)
    plt.xlabel('Fibre #')
    plt.ylabel('RV [m/s]')
#     plt.legend()

#     ax3 = fig.add_subplot(322) #Histogram
    ax3 = fig.add_subplot(gs[2,0:2]) #Histogram
    hist = np.histogram(RV[~np.isnan(RV)],50)
    ax3.bar(hist[1][1:],hist[0], width = (hist[1][-2]-hist[1][-1]), color = 'k', alpha = 0.5)
    ax3.set_ylabel('Counts')
    ax3.set_xlabel('RV [m/s]')

    
    
    ax4 = ax3.twinx()
#     # ax4.scatter(RV,SNR, c='b', s=50)
    ax4.plot(RV, SNR, 'o', color = 'b', picker=5)
    ax4.plot([baryVels[epoch], baryVels[epoch]], [0, maxSNRs[cam]], 'g', lw=2)
    ax4.plot([0, 0], [0, maxSNRs[cam]], 'k--', lw=2)
    ax4.set_ylabel('SNR')
    ax4.set_ylim(0, maxSNRs[cam])

#     ax5= fig.add_subplot(324) #RVs
    ax5= fig.add_subplot(gs[:,2:]) #RVs
    ax5.plot(RA, DEC, 'o', label = 'RV', color = 'b', picker=5)
    ax5.grid(True)
    plt.xlabel('Dec [deg]')
    plt.ylabel('RA [deg]')

    plt.tight_layout()

    fig.canvas.mpl_connect('pick_event', onpick1)

    plt.show()
    
else:
    print 'No arguments supplied. Call run_multiplot [cam] [epoch]'


# <codecell>



# labels = ['Blue','Green','Red','IR']


# data=np.load('npy/data.npy')
# RVs=np.load('npy/RVs.npy')
# baryVels=np.load('npy/baryVels.npy')
# SNRs = np.load('npy/SNRs.npy')
# #     JDs=np.load('npy/JDs.npy')
# #     sigmas=np.load('npy/sigmas.npy')

# FIBRE = data[:,2].astype(float) #fibres
# RV = RVs[:,epoch,cam] #RV
# SNR = SNRs[:,epoch,cam] #SNR
# # baryVel = baryVels[:,epoch,cam] #baryVels
# maxSNRs = np.nanmax(SNRs,axis=(0,1))


# fig = plt.figure()

# ax1 = fig.add_subplot(311) #SNRs
# ax1.scatter(FIBRE, SNR, label = 'SNR', color = 'b')
# plt.xlabel('Fibre #')
# plt.ylabel('SNR')
# plt.legend()

# ax2= fig.add_subplot(312) #RVs
# ax2.scatter(FIBRE, RV, label = 'RV', color = 'b')
# plt.xlabel('Fibre #')
# plt.ylabel('RV [m/s]')
# plt.legend()

# ax3 = fig.add_subplot(313) #Histogram
# ax3.bar(hist[1][1:],hist[0], width = (hist[1][-2]-hist[1][-1]), color = 'k', alpha = 0.5)
# ax3.set_ylabel('Counts')
# ax3.set_xlabel('RV [m/s]')

# ax4 = ax3.twinx()
# ax4.scatter(RV,SNR, c='b', s=50)
# ax4.plot([baryVels[epoch], baryVels[epoch]], [0, maxSNRs[cam]], 'r', lw=2)
# ax4.plot([0, 0], [0, maxSNRs[cam]], 'k--', lw=2)
# ax4.set_ylabel('SNR')
# ax4.set_ylim(0, maxSNRs[cam])


# plt.tight_layout()


# plt.show()


# <codecell>

# import numpy as np
# import pylab as plt



# RVs = np.load('npy/RVs.npy')
# SNRs = np.load('npy/SNRs.npy')
# baryVels = np.load('npy/baryVels.npy')

 
# cameras = ['Blue','Green','Red','IR']
# epoch = 7 
# cam = 0

# maxSNRs = np.nanmax(SNRs,axis=(0,1))
 


        
        
# RV = RVs[:,epoch,cam]
# SNR = SNRs[:,epoch,cam]
# hist = np.histogram(RV,50)
        
# fig = plt.figure()
    
# ax = fig.add_subplot(111)
# ax.bar(hist[1][1:],hist[0], width = (hist[1][-2]-hist[1][-1]))
# ax.set_ylabel('Counts')
# ax.set_xlabel('RV [m/s]')

# ax2 = ax.twinx()
# ax2.scatter(RV,SNR, c='r', s=100)
# ax2.plot([baryVels[epoch], baryVels[epoch]], [0, maxSNRs[cam]], 'red', lw=2)
# ax2.plot([0, 0], [0, maxSNRs[cam]], 'k--', lw=2)
# ax2.set_ylabel('SNR')
# ax2.set_ylim(0, maxSNRs[cam])
# plt.show()

# <codecell>

# from numpy.random import rand
# import matplotlib
# # matplotlib.use('gtkagg')
# import matplotlib.pyplot as plt

# # create all axes we need
# ax0 = plt.subplot(211)
# ax1 = ax0.twinx()
# ax2 = plt.subplot(212)
# ax3 = ax2.twinx()

# # share the secondary axes
# ax1.get_shared_y_axes().join(ax1, ax3)

# ax0.plot(rand(1) * rand(10),'r')
# ax1.plot(10*rand(1) * rand(10),'b')
# ax2.plot(3*rand(1) * rand(10),'g')
# ax3.plot(10*rand(1) * rand(10),'y')
# plt.show()

