#!/opt/local/bin/python

import os
import create_obj as cr_obj
import pickle
import glob
import pyfits as pf
import numpy as np
import RVTools as RVT
import sys

CCReferenceSet = 0 
medianRange = 0
useRangeFilter = False
minMJD = 0

if len(sys.argv)>1:
    CCReferenceSet = int(sys.argv[1])

if len(sys.argv)>2:
    fileList = [sys.argv[2]]
else:
    fileList = glob.glob('obj/*.obj')

if len(fileList)>0:
    i=0
    for filename in fileList[:]:
        if 'red' not in filename:
            print filename
            filehandler = open(filename, 'r')
            thisStar = pickle.load(filehandler)
    #         try:
    #         RVT.find_max_wl_range(thisStar)
            if thisStar.type=='arc':
                print 'Arc'
                RVT.RVs_CC_t0_arc(thisStar, corrHWidth = 5)
                
            elif thisStar.type=='star':
                print 'Star'
                RVT.RVs_CC_t0(thisStar,i, minMJD=minMJD, CCReferenceSet=CCReferenceSet, medianRange=medianRange, useRangeFilter = useRangeFilter)
            i+=1
    
            file_pi = open('obj/red_'+thisStar.name+'.obj', 'w') 
            pickle.dump(thisStar, file_pi) 
            file_pi.close()
    #         except:
    #             print 'CC failed (probably NaNs)'
            thisStar = None
            print ''
else:
    print 'No .obj files found in obj/'
    
    
