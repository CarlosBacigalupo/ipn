#!/opt/local/bin/python

import os
import create_obj as cr_obj
import pickle
import glob
import pyfits as pf
import numpy as np
import sys
import toolbox
import importlib

booHD1581 = False
IRAFFiles = '/Users/Carlos/Documents/workspace/GAP/IrafReduction/results/'   #folder to IRAF reduced files

if len(sys.argv)>1:

    try: 
        os.mkdir('cam1')
        os.mkdir('cam2')
        os.mkdir('cam3')
        os.mkdir('cam4')
        os.mkdir('obj')
    except:
        pass

    dataset = sys.argv[1]
    try:
        thisDataset = importlib.import_module('data_sets.'+dataset)
    except:
        print 'Could not load',dataset         
        sys.exit()


    #compose file prefixes from date_list
    months = np.array(['', 'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
    d = np.array([s[4:] for s in thisDataset.date_list])
    m = months[np.array([s[2:4] for s in thisDataset.date_list]).astype(int)]
    filename_prfx = np.core.defchararray.add(d, m)

    #copy files
    for i,folder in enumerate(thisDataset.date_list):
        for files in thisDataset.ix_array[i][2:]:
            for cam in range(1,5):
                strCopy = 'cp ' + IRAFFiles + folder + '/norm/' + filename_prfx[i] + str(cam) + "%04d" % (files,) + '.ms.fits ' 
                strCopy += 'cam'+ str(cam) + '/' + filename_prfx[i] + str(cam) + "%04d" % (files,) + '.fits ' 
                print strCopy
                try:
                    os.system(strCopy)
                except:
                    print 'no copy'
                
else:
    print 'arguments missing'
    
                   
sys.exit(0)

#load all star names from 1st file
fileList = glob.glob('cam1/*.fits')
starNames = []
if len(fileList)>0:
    if len(sys.argv)>1:
        starNames = [sys.argv[1]]
    else:
        for fitsname in fileList[:]:
            print "Starnames",starNames, 'file', fitsname
            fits = pf.open(fitsname)
            a = fits['FIBRES'].data            
            starNames = np.hstack((starNames,np.array(a.field('NAME')[a.field('TYPE').strip()=='P'])))
            fits.close()
    
        starNames = np.unique(starNames)
    
    if booHD1581==True: starNames = ['Giant01']
    
    print 'Collecting data from ',len(starNames),'stars'
    print starNames
    for i,star_name in enumerate(starNames):
        print i,star_name
        thisStar = cr_obj.star(star_name)
          
        thisStar.exposures = cr_obj.exposures()
        thisStar.exposures.load_exposures(thisStar.name)
         
        if booHD1581==True: #to fix wrong values due to offset field
            thisStar.RA = 5.017749791666667   #from simbad
            thisStar.Dec = -64.87479302777777 #from simbad
            thisStar.RA_deg, thisStar.RA_min, thisStar.RA_sec = toolbox.dec2sex(thisStar.RA)   
            thisStar.Dec_deg, thisStar.Dec_min, thisStar.Dec_sec = toolbox.dec2sex(thisStar.Dec)
                         
        thisStar.exposures.calculate_baryVels(thisStar)
        if booHD1581==True: star_name = 'HD1581'
        thisStar.name = star_name
        file_pi = open('obj/'+star_name+'.obj', 'w') 
        pickle.dump(thisStar, file_pi) 
        file_pi.close()
else:
    print 'No files found in cam1/*.fits'