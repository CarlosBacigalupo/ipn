#!/usr/bin/python
import pyfits as pf
import numpy as np
import pylab as plt
import os
import glob
import pickle
import RVTools as RVT
import toolbox
import sys

if len(sys.argv)>1:
    objects = sys.argv[1].split(',')
    
    print objects
    
    allFiles = np.genfromtxt('all_data.txt', dtype=str)
    
    # raw = []
    print 'Obs & Filename & Field Name & Plate & MJD & Relative Day (days) & Exp Time (s) \\\\'
    
    # good_pivots = [175,175,224]
    
    
    outFileNames = []
    outPlates = []
    outObjects = []
    outMJD_Exp = []
    #t0 = 0  
    #order = 0 
    #dirs = glob.glob('*')
    for thisFile in allFiles:
        a = pf.open(thisFile)
        if a[0].header['NDFCLASS'] == 'MFOBJECT':
            print 'class object',a[0].header['OBJECT']
            if a[0].header['OBJECT'] in objects:
                print 'OBJECT'
                #pivot_idx = np.where(a[0].header['OBJECT']== objects)[0][0]
		#print np.where(a[0].header['OBJECT']== objects)
                e = float(a[0].header['EXPOSED'])/2/24/60/60 # EXPOSED/2 in days
                inDate = np.hstack((a[0].header['UTDATE'].split(':'),a[0].header['UTSTART'].split(':'))).astype(int)
                UTMJD = toolbox.gd2jd(inDate, TZ=0)-2400000.5 + e
		outFileNames.append(thisFile.split('/')[-1])
		outPlates.append(a[0].header['SOURCE'])
		outObjects.append(a[0].header['OBJECT'])
		outMJD_Exp.append([UTMJD, a[0].header['EXPOSED']])

   #             if t0==0:t0=UTMJD
   #             tDiff=UTMJD-t0
    #                 print str(order)+' & '+ i+' & HD1581 & '+a[0].header['SOURCE']+' & '+str(UTMJD)+' & '+str(tDiff)+' & '+str(a[0].header['EXPOSED']) +' \\\\'
    #                 print str(order)+' & '+ i.split('/')[-1]+' & HD285507 & '+a[0].header['SOURCE']+' & '+str(UTMJD)+' & '+str(tDiff)+' & '+str(a[0].header['EXPOSED']) +' \\\\'
    #            print str(order)+' & '+ thisFile.split('/')[-1]+' & '+a[0].header['OBJECT']+' & '+a[0].header['SOURCE']+' & '+str(UTMJD)+' & '+str(tDiff)+' & '+str(a[0].header['EXPOSED']) +' \\\\'
    #                 print str(order)+' & '+ i.split('/')[-1]+' & '+a[0].header['OBJECT']+' & '+a[0].header['SOURCE']+' & '+str(UTMJD)+' & '+str(tDiff)+' & '+str(a[0].header['EXPOSED']) +' \\\\'
     #           order +=1
    #                 raw.append((i, a[0].header['OBJECT'],a[0].header['SOURCE'],a[0].header['UTMJD'],a[0].header['EXPOSED']))            
        a.close
    print outFileNames, outPlates, outObjects, outMJD_Exp
    outFileNames = np.array(outFileNames)
    outPlates = np.array(outPlates)
    outObjects = np.array(outObjects)
    outMJD_Exp = np.array(outMJD_Exp)
    
    np.save('npy/outFileNames.npy',outFileNames)
    np.save('npy/outPlates.npy',outPlates)
    np.save('npy/outObjects.npy',outObjects)
    np.save('npy/outMJD_Exp.npy',outMJD_Exp)
	 
    # raw = np.array(raw)
else:
    print 'No objects specified.'


