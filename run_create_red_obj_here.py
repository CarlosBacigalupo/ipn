
import os
import create_obj as cr_obj
import pickle
import glob
import pyfits as pf
import numpy as np
import RVTools as RVT
import sys

sys.path = ['', '/usr/local/yt-hg', '/home/science/staff/kalumbe/my-astro-lib', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', 
           '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', 
           '/usr/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages/PILcompat', '/usr/lib/python2.7/dist-packages/gtk-2.0', 
           '/usr/lib/pymodules/python2.7', 
           '/usr/lib/python2.7/dist-packages/ubuntu-sso-client', '/usr/lib/python2.7/dist-packages/ubuntuone-client', 
           '/usr/lib/python2.7/dist-packages/ubuntuone-couch', '/usr/lib/python2.7/dist-packages/ubuntuone-storage-protocol', 
           '/usr/lib/python2.7/dist-packages/wx-2.8-gtk2-unicode']

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
    
    
