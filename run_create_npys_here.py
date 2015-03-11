
import os
import create_obj as cr_obj
import pickle
import glob
import pyfits as pf
import numpy as np
import RVTools as RVT
import sys


#load all star names from 1st file
fileList = glob.glob('red_*.obj')

if len(fileList)>0:


    for i,filename in enumerate(fileList):
        print i,filename,

        filehandler = open(filename, 'r')
        thisStar = pickle.load(filehandler)

        if i==0:
            nObs = thisStar.exposures.JDs.shape[0]
            print 'Found', nObs, 'observations'
            data = []
            RVs = np.zeros((len(fileList),nObs,4))
            sigmas = np.ones((len(fileList),nObs,4)) * 100

#         if ((thisStar.exposures.cameras[0].sigmas.all()!=0) and
#             (thisStar.exposures.cameras[1].sigmas.all()!=0) and
#             (thisStar.exposures.cameras[2].sigmas.all()!=0) and
#             (thisStar.exposures.cameras[3].sigmas.all()!=0)):
            
        print 'Appending...',
        data.append([thisStar.name, thisStar.Vmag,np.unique(thisStar.exposures.pivots)[0]])
        RVs[i,:,0] = thisStar.exposures.cameras[0].RVs
        RVs[i,:,1] = thisStar.exposures.cameras[1].RVs
        RVs[i,:,2] = thisStar.exposures.cameras[2].RVs
        RVs[i,:,3] = thisStar.exposures.cameras[3].RVs
        sigmas[i,:,0] = thisStar.exposures.cameras[0].sigmas
        sigmas[i,:,1] = thisStar.exposures.cameras[1].sigmas
        sigmas[i,:,2] = thisStar.exposures.cameras[2].sigmas
        sigmas[i,:,3] = thisStar.exposures.cameras[3].sigmas
        JDs = np.array(thisStar.exposures.JDs)
        filehandler.close()
        thisStar = None
        
#         else:
#             print 'Found sigma=0'
        
        print ''
    
    
    data = np.array(data)
    order = np.argsort(data[:,2].astype(float).astype(int))
    
#     data = data[order]
#     RVs = RVs[order]
#     sigmas = sigmas[order]
#     
    #save?
    np.save('data',data)
    np.save('RVs',RVs)
    np.save('sigmas',sigmas)
    np.save('JDs',JDs)

    print ''
    print 'data',len(data)
    print 'RVs',RVs.shape
    print 'sigmas',sigmas.shape
    print 'JDs',JDs.shape

else:
    print 'No reduced .obj files found here'
    
    
