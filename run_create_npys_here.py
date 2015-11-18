#!/opt/local/bin/python

import os
import create_obj as cr_obj
import pickle
import glob
import pyfits as pf
import numpy as np
import RVTools as RVT
import sys

try: 
    os.mkdir('npy')
except:
    pass


#load all star names from 1st file
fileList = glob.glob('obj/red_*.obj')

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
            SNRs = np.zeros((len(fileList),nObs,4))
            sigmas = np.ones((len(fileList),nObs,4)) * 100

#         if ((thisStar.exposures.cameras[0].sigmas.all()!=0) and
#             (thisStar.exposures.cameras[1].sigmas.all()!=0) and
#             (thisStar.exposures.cameras[2].sigmas.all()!=0) and
#             (thisStar.exposures.cameras[3].sigmas.all()!=0)):
            
        print 'Appending...',
        data.append([thisStar.name, thisStar.Vmag,np.unique(thisStar.exposures.pivots)[0], thisStar.RA, thisStar.Dec])
        RVs[i,:,0] = thisStar.exposures.cameras[0].RVs
        RVs[i,:,1] = thisStar.exposures.cameras[1].RVs
        RVs[i,:,2] = thisStar.exposures.cameras[2].RVs
        RVs[i,:,3] = thisStar.exposures.cameras[3].RVs
        SNRs[i,:,0] = thisStar.exposures.cameras[0].SNRs
        SNRs[i,:,1] = thisStar.exposures.cameras[1].SNRs
        SNRs[i,:,2] = thisStar.exposures.cameras[2].SNRs
        SNRs[i,:,3] = thisStar.exposures.cameras[3].SNRs
        sigmas[i,:,0] = thisStar.exposures.cameras[0].sigmas
        sigmas[i,:,1] = thisStar.exposures.cameras[1].sigmas
        sigmas[i,:,2] = thisStar.exposures.cameras[2].sigmas
        sigmas[i,:,3] = thisStar.exposures.cameras[3].sigmas
        MJDs = np.array(thisStar.exposures.JDs)
        baryVels = np.array(thisStar.exposures.abs_baryVels-thisStar.exposures.abs_baryVels[0])
        filehandler.close()
        thisStar = None
        
#         else:
#             print 'Found sigma=0'
        
        print ''
    
    baryVels3D = np.zeros(RVs.shape)
    baryVels3D[:,:,0] = np.tile(baryVels,[RVs.shape[0],1])
    baryVels3D[:,:,1] = np.tile(baryVels,[RVs.shape[0],1])
    baryVels3D[:,:,2] = np.tile(baryVels,[RVs.shape[0],1])
    baryVels3D[:,:,3] = np.tile(baryVels,[RVs.shape[0],1])
    RVs[RVs==0.]=np.nan
    baryRVs = RVs - baryVels3D

    
    data = np.array(data)
    order = np.argsort(data[:,2].astype(float).astype(int))
    
    data = data[order]
    RVs = RVs[order]
    baryRVs = baryRVs[order]
    SNRs = SNRs[order]
    sigmas = sigmas[order]

#     
    #save?
    np.save('npy/data',data)
    np.save('npy/RVs',RVs)
    np.save('npy/baryRVs',baryRVs)
    np.save('npy/SNRs',SNRs)
    np.save('npy/sigmas',sigmas)
    np.save('npy/baryVels',baryVels)
    np.save('npy/MJDs',MJDs)

    print ''
    print 'data',data.shape
    print 'RVs',RVs.shape
    print 'baryRVs',baryRVs.shape
    print 'SNRs',SNRs.shape
    print 'sigmas',sigmas.shape
    print 'baryVels',baryVels.shape
    print 'MJDs',MJDs.shape

else:
    print 'No reduced .obj files found here'
    
    
