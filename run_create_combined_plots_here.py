
import os
import pickle
import glob
import pyfits as pf
import numpy as np
import sys
import pickle
import RVTools as RVT
import pylab as plt

folder1 = '/Users/Carlos/Documents/HERMES/reductions/47Tuc_core_6.2'
folder2 = '/Users/Carlos/Documents/HERMES/reductions/47Tuc_core_1arc_6.2'

folder1 = '/Users/Carlos/Documents/HERMES/reductions/m67_6.2'
folder2 = '/Users/Carlos/Documents/HERMES/reductions/m67_1arc_6.2'



# if len(sys.argv)>1:
#     fileList = [sys.argv[1]]
# else:
#     fileList = glob.glob('*.obj')
# 
# try:os.makedirs('plots')
# except:pass

RVs1 = np.load(folder1+'/RVs.npy')
RVs2 = np.load(folder2+'/RVs.npy')

RVs = RVs1-RVs2

os.system('cp '+folder1+'/data.npy .')
os.system('cp '+folder1+'/sigmas.npy .')
os.system('cp '+folder1+'/JDs.npy .')
os.system('cp '+folder1+'/baryVels.npy .')
np.save('RVs', RVs)

# 
# if len(fileList)>0:
#     for objName in fileList:
#         if 'red' not in objName:
#             print 'Reading',objName,
#             filehandler = open(objName, 'r')
#             thisStar = pickle.load(filehandler)
# 
#             for cam, thisCam in enumerate(thisStar.exposures.cameras):
#                 for x,y,label,i in zip(thisCam.wavelengths, thisCam.red_fluxes, thisCam.fileNames, range(thisCam.wavelengths.shape[0])):
#                     if np.nansum(x)!=0: 
#                         __,y1 = RVT.clean_flux(x,y,thisCam)
#                         plt.plot(x,y1+i, label= label, c='k')
#                 plt.title(thisStar.name + ' cam' + str(cam+1))
#                 plt.yticks = thisCam.fileNames 
#                 plt.gca().set_yticks(range(thisCam.wavelengths.shape[0]))
#                 plt.gca().set_yticklabels(thisCam.fileNames)
#                 plt.tight_layout()
#                 plt.savefig('plots/' + objName[:-4] + '_fluxes_cam' + str(cam+1), dpi = 1000 )
#                 print cam+1,
# #                 plt.show()    
#                 plt.close()
#             print ''
# else:
#     print 'No *.obj files here. (red_files.obj are ignored)'
#     
