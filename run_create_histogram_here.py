
import os
import pickle
import glob
# import pyfits as pf
import numpy as np
import sys
import RVPlots as RVP
import pylab as plt



booSave = True
booShow = False
booShowBary = True


if len(sys.argv)>1:
    maxRV = np.absolute(float(sys.argv[1]))
else:
    maxRV = 5000
minRV = -maxRV

if len(sys.argv)>2:
    baryOffset = bool(sys.argv[2])
else:
    baryOffset = False
print baryOffset   

try:
    os.makedirs('plots')
    
except:
    print 'Falied to create plots/ folder'
    print ''

try:
    os.makedirs('plots/1')
    os.makedirs('plots/2')
    os.makedirs('plots/3')
    os.makedirs('plots/4')
except:
    pass


RVs = np.load('npy/RVs.npy')
SNRs = np.load('npy/SNRs.npy')
baryVels = np.load('npy/baryVels.npy')

 
cameras = ['Blue','Green','Red','IR']

# print np.max(RVs,axis=(0,1)),np.max(SNRs,axis=(0,1))
# maxRVs = np.max(RVs,axis=(0,1))
maxSNRs = np.nanmax(SNRs,axis=(0,1))
 


for epoch in range(RVs.shape[1])[:]:
    if baryOffset==True:
        offset = baryVels[epoch]
    else:
        offset = 0
        
    for cam in range(4)[:]:
        print 'Plotting epoch,cam',epoch,cam
        print '    Limits set to max,min', minRV+offset, maxRV+offset
        
        
        R = RVs[:,epoch,cam]
        filter = ((R>(minRV+offset)) & (R<(maxRV+offset)))
        S = SNRs[:,epoch,cam][filter]
        R = R[filter]
        hist = np.histogram(R,50)
        
        title = os.getcwd().split('/')[-1]+ ' - t'+str(epoch)+', '+cameras[cam]+' camera'
        plotName = 'plots/'+str(cam+1)+'/hist_'+str(epoch)
        fig = plt.figure()
        plt.title(title)
    
        ax = fig.add_subplot(111)
        ax.bar(hist[1][1:],hist[0], width = (hist[1][-2]-hist[1][-1]))
#         ax.grid()
        ax.set_ylabel('Counts')
        ax.set_xlabel('RV [m/s]')
#         ax.set_xlim(baryVels[epoch]-minRV,baryVels[epoch]+maxRV)
#         ax.set_ylim(0,10)

        ax2 = ax.twinx()
        ax2.scatter(R,S, c='r', s=100)
        if booShowBary==True: ax2.plot([baryVels[epoch], baryVels[epoch]], [0, maxSNRs[cam]], 'red', lw=2)
#             ax2.bar(baryVels[epoch],maxSNRs[cam] ,color = 'green', lw=1)
#         ax2.grid()
        ax2.set_ylabel('SNR')
        ax2.set_ylim(0, maxSNRs[cam])
        ax2.set_xlim(minRV+offset,maxRV+offset)
        ax2.plot([0, 0], [0, maxSNRs[cam]], 'k--', lw=2)
        
        if booSave==True:plt.savefig(plotName)    
        if booShow==True: plt.show()
        plt.close()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('mathtext', default='regular')


#Plot
# RVP.RVs_all_stars()

#Plot
# RVP.RVs_all_stars_NPYs(RVClip = -1, booSave=booSave , booBaryPlot=booShow, title = 'M67 1arc Diff')


# if len(fileList)>0:
#     for objName in fileList[:2]:
#         print 'Reading',objName
#         filehandler = open(objName, 'r')
#         thisStar = pickle.load(filehandler)
        
        #Plot
#         RVP.all_spec_overlap(thisStar, booSave = booSave,booShow = booShow )
        
        #Plot
#         RVP.RVs_single_star(thisStar, sigmaClip = -1, RVClip = 3000)

            
#         thisStar = None
#         print ''
# else:
#     print 'No red_*.obj files here.'
    
