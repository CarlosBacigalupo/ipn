
import os
import create_obj as cr_obj
import pickle
import glob
import pyfits as pf
import numpy as np
import sys
import toolbox

booHD1581 = False

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
    
    for i,star_name in enumerate(starNames):
        print i,star_name
        thisStar = cr_obj.star(star_name)
         
        thisStar.exposures = cr_obj.exposures()
        thisStar.exposures.load_exposures(thisStar.name)
        
        if booHD1581==True: #to fix wrong values due to offset field
            thisStar.RA_dec = 0.087738428718276612
            thisStar.Dec_dec = -1.1321689058299416       
            thisStar.RA_h, thisStar.RA_min, thisStar.RA_sec = toolbox.dec2sex(np.rad2deg(thisStar.RA_dec/15))   
            thisStar.Dec_deg, thisStar.Dec_min, thisStar.Dec_sec = toolbox.dec2sex(np.rad2deg(thisStar.Dec_dec))
                        
        thisStar.exposures.calculate_baryVels(thisStar)
        if booHD1581==True: star_name = 'HD1581'
        thisStar.name = star_name
        file_pi = open(star_name+'.obj', 'w') 
        pickle.dump(thisStar, file_pi) 
        file_pi.close()
else:
    print 'No files found in cam1/*.fits'