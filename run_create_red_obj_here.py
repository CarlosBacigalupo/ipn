
import os
import create_obj as cr_obj
import pickle
import glob
import pyfits as pf
import numpy as np
import RVTools as RVT
import sys

CCReferenceSet = 0 
if len(sys.argv)>1:
    CCReferenceSet = int(sys.argv[1])

if len(sys.argv)>2:
    fileList = [sys.argv[2]]
else:
    #load all star names from 1st file
    fileList = glob.glob('*.obj')

if len(fileList)>0:
    for filename in fileList    :
        if 'red' not in filename:
            print filename
            filehandler = open(filename, 'r')
            thisStar = pickle.load(filehandler)
    #         try:
    #         RVT.find_max_wl_range(thisStar)
            RVT.RVs_CC_t0(thisStar, CCReferenceSet=CCReferenceSet)
    
            file_pi = open('red_'+thisStar.name+'.obj', 'w') 
            pickle.dump(thisStar, file_pi) 
            file_pi.close()
    #         except:
    #             print 'CC failed (probably NaNs)'
            thisStar = None
            print ''
else:
    print 'No .obj files found here'
    
    
