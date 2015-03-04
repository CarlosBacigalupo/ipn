
import os
import create_obj as cr_obj
import pickle
import glob
import TableBrowser as tb
import pyfits as pf


#load all star names from 1st file
fileList = glob.glob('cam1/*.fits')
if len(fileList)>0:
    fits = pf.open(fileList[0])
    a = fits['FIBRES'].data            
    starNames=a.field('NAME')[a.field('TYPE').strip()=='P']
    
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