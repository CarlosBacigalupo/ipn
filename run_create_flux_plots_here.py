#!/opt/local/bin/python

import os
import pickle
import glob
import pyfits as pf
import numpy as np
import sys
import pickle
import RVTools as RVT
import pylab as plt

if ((len(sys.argv)>1) and ('obj' in sys.argv[1])):
    fileList = [sys.argv[1]]
else:
    fileList = glob.glob('obj/*.obj')

if len(sys.argv)>2:
    cameras = np.array([sys.argv[2]])
else:
    cameras = np.arange(4)

try:os.makedirs('plots')
except:pass

if len(fileList)>0:
    for objName in fileList:
        if 'red' not in objName:
            print 'Reading',objName,
            filehandler = open(objName, 'r')
            thisStar = pickle.load(filehandler)

            for cam, thisCam in enumerate(thisStar.exposures.cameras):
                if cam in cameras.astype(int):
                    for x,y,label,i in zip(thisCam.wavelengths, thisCam.red_fluxes, thisCam.fileNames, range(thisCam.wavelengths.shape[0])):
                        if np.nansum(x)!=0: 
                            __,y1 = RVT.clean_flux(x,y,thisCam,medianRange=5)
                            plt.plot(x,y1+i, label= label, c='k')
                    plt.title(thisStar.name + ' cam' + str(cam+1))
                    plt.yticks = thisCam.fileNames 
                    plt.gca().set_yticks(range(thisCam.wavelengths.shape[0]))
                    plt.gca().set_yticklabels(thisCam.fileNames)
    #                 plt.tight_layout()
                    plt.savefig('plots/' + objName[4:-4] + '_fluxes_cam' + str(cam+1), dpi = 500 )
                    print cam+1,
#                     plt.show()    
                    plt.close()
            print ''
            thisStar = None
            filehandler.close()
else:
    print 'No *.obj files here. (red_files.obj are ignored)'
    
