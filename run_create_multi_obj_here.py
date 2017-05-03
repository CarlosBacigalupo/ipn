#!/usr/bin/python

import os
import create_obj as cr_obj
import pickle
import glob
import pyfits as pf
import numpy as np
import sys
import toolbox

# booHD1581 = False
# if os.getcwd().split('/')[-1]=='HD1581': booHD1581 = True

try: 
    os.mkdir('obj')
except:
    pass


#load all star names from starNames_piv_pl_mjd
# fileList = glob.glob('cam1/*.fits')
starNames = np.load('npy/starNames_piv_pl_mjd.npy')
# starNames = ['M67-375_216_1_56668.5921065']

if len(starNames)>0:
#     if len(sys.argv)>1:
#         starNames = [sys.argv[1]]
#     else:
#         for fitsname in fileList[:]:
#             print "Starnames",starNames, 'file', fitsname
#             fits = pf.open(fitsname)
#             a = fits['FIBRES'].data            
#             starNames = np.hstack((starNames,np.array(a.field('NAME')[a.field('TYPE').strip()=='P'])))
#             fits.close()
#     
#         starNames = np.unique(starNames)
#     
#     if booHD1581==True: starNames = ['Giant01']
    
    print 'Collecting data from ',len(starNames),'starNames_piv_pl_mjd'
    print 
    
    for i,combinedName in enumerate(starNames[:1]):
        starName, pivot, plate, MJD = combinedName.split('_')
        pivot = int(pivot)
        plate = int(plate)
        MJD = float(MJD)
#         print i,combinedName,starName, pivot, plate, MJD
        thisStar = cr_obj.star(starName)
        thisStar.type='star'  
        thisStar.exposures = cr_obj.exposures()
        thisStar.exposures.load_multi_exposures(starName, pivot, plate, MJD)
         
#         if booHD1581==True: #to fix wrong values due to offset field
#             thisStar.RA = 5.017749791666667   #from simbad
#             thisStar.Dec = -64.87479302777777 #from simbad
#             thisStar.RA_deg, thisStar.RA_min, thisStar.RA_sec = toolbox.dec2sex(thisStar.RA)   
#             thisStar.Dec_deg, thisStar.Dec_min, thisStar.Dec_sec = toolbox.dec2sex(thisStar.Dec)
#                          
        thisStar.exposures.calculate_baryVels(thisStar)
#         if booHD1581==True: star_name = 'HD1581'
#         thisStar.name = star_name
        file_pi = open('obj/'+combinedName+'.obj', 'w') 
        pickle.dump(thisStar, file_pi) 
        file_pi.close()
else:
    print 'No files found in cam1/*.fits'