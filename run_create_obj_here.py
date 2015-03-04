
import os
import create_obj as cr_obj
import pickle
import glob
import pyfits as pf
import numpy as np


#load all star names from 1st file
fileList = glob.glob('cam1/*.fits')
starNames = []
if len(fileList)>0:
    for fitsname in fileList[:]:
        print "Starnames",starNames, 'file', fitsname
        fits = pf.open(fitsname)
        a = fits['FIBRES'].data            
        starNames = np.hstack((starNames,np.array(a.field('NAME')[a.field('TYPE').strip()=='P'])))
        fits.close()

    starNames = np.unique(starNames)
    print 'Collecting data from ',len(starNames),'stars'
    
    for i,star_name in enumerate(starNames):
        print i,star_name
        thisStar = cr_obj.star(star_name)
         
        thisStar.exposures = cr_obj.exposures()
        thisStar.exposures.load_exposures(thisStar.name)
        thisStar.exposures.calculate_baryVels(thisStar)
        thisStar.name = star_name
        file_pi = open(star_name+'.obj', 'w') 
        pickle.dump(thisStar, file_pi) 
        file_pi.close()
else:
    print 'No files found in cam1/*.fits'