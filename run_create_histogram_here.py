
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

if len(sys.argv)>1:
    fileList = [sys.argv[1]]

else:
    fileList = glob.glob('red_*.obj')

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


RVs = np.load('RVs.npy')
SNRs = np.load('SNRs.npy')

cameras = ['Blue','Green','Red','IR']

# print np.max(RVs,axis=(0,1)),np.max(SNRs,axis=(0,1))
# maxRVs = np.max(RVs,axis=(0,1))
maxSNRs =np.max(SNRs,axis=(0,1))
 


for epoch in range(RVs.shape[1])[:]:
    for cam in range(4)[:]:
        print 'Plotting epoch,cam',epoch,cam
        R = RVs[:,epoch,cam]
        S = SNRs[:,epoch,cam]
        
        a = np.histogram(R)
        title = os.getcwd().split('/')[-1]+ ' - t'+str(epoch)+', '+cameras[cam]+' camera'
        plotName = 'plots/'+str(cam+1)+'/hist_'+str(epoch) 
        
        fig = plt.figure()
        plt.title(title)
    
        ax = fig.add_subplot(111)
        ax.bar(a[1][1:],a[0], width = (a[1][-2]-a[1][-1]))
        ax2 = ax.twinx()
        ax2.scatter(R,S, c='r', s=100)
#         ax.legend(loc=0)
        ax.grid()
        ax2.grid()
        ax.set_ylabel('Counts')
        ax.set_xlabel('RV [m/s]')
    
#         ax.set_ylim(0,10)
        ax2.set_ylim(0, maxSNRs[cam])
    #     plt.show()
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
    
