#!/opt/local/bin/python

import os
import pickle
import glob
# import pyfits as pf
# import numpy as np
import sys
import RVPlots as RVP
# import pylab as plt



booSave = True     
booShow = False
booBaryPlot = False
thisCamIdx = -1
RVClip = 2000
topStars = -1 #how many stars to read. -1 for all.
booBaryCorrect = True
idStars = ['Giant01'] #name of stars to id in RV plot
idStars = ['HD1581'] #name of stars to id in RV plot
booShowArcRVs = True
booFit = True

if len(sys.argv)>1:
    fileList = [sys.argv[1]]

else:
    fileList = glob.glob('obj/red_*.obj')

try:
    os.makedirs('plots')
except:
    pass

try:
    os.makedirs('plots/1')
    os.makedirs('plots/2')
    os.makedirs('plots/3')
    os.makedirs('plots/4')
except:
    pass

#Plot
# RVP.RVs_all_stars()

#Plot
RVP.RVs_all_stars_NPYs(booShowArcRVs = booShowArcRVs, idStars= idStars, sigmaClip = -1,RVClip = RVClip, topStars=topStars, booSave = booSave,booShow = booShow  , booBaryPlot=booBaryPlot, booBaryCorrect = booBaryCorrect)

#Plot
# RVP.RVs_by_star_NPYs(RVClip = RVClip, booSave = booSave,booShow = booShow )

#Plot
# RVP.SNR_RV_vs_fibre(RVClip = RVClip, booSave = booSave,booShow = booShow)
 
#Plot
# RVP.RV_vs_fibre(RVClip = RVClip, booSave = booSave,booShow = booShow)

#Plot
# RVP.flux_and_CC(RVref = RVClip, booSave = booSave, booShow = booShow)

#Plot
# RVP.RVCorr_RV(RVClip = RVClip, booSave = booSave, booShow = booShow, booBaryPlot=booBaryPlot)

#Plot
# RVP.RVCorr_Slit(RVClip = RVClip, booSave = booSave, booShow = booShow, booBaryPlot=booBaryPlot)

#Plot
# RVP.arcRVs(booSave = booSave, booShow = booShow, booFit = booFit)


  

# if len(fileList)>0:
#     for objName in fileList[:]:
#         print 'Reading',objName
#         filehandler = open(objName, 'r')
#         thisStar = pickle.load(filehandler)
#               
#         #Plot
#         RVP.all_spec_overlap(thisStar, booSave = booSave, thisCamIdx = thisCamIdx, booShow = booShow )
#    
#         #Plot
# #         RVP.RVs_single_star(thisStar, sigmaClip = -1, RVClip = 3000)
#       
#                   
#         thisStar = None
#         filehandler.close()
#         print ''
# else:
#     print 'No red_*.obj files here.'
       
