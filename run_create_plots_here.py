
import os
import pickle
import glob
# import pyfits as pf
# import numpy as np
import sys
import RVPlots as RVP
# import pylab as plt



booSave = False
booShow = True

if len(sys.argv)>1:
    fileList = [sys.argv[1]]

else:
    fileList = glob.glob('red_*.obj')

try:
    os.makedirs('plots')
except:
    print 'Falied to create plots/'
    print ''

#Plot
# RVP.RVs_all_stars()

#Plot
RVP.RVs_all_stars_NPYs(RVClip = -1, booSave=True, booBaryPlot=False, title = 'HD1581 - 1arc differences')


# if len(fileList)>0:
#     for objName in fileList[:2]:
#         print 'Reading',objName
#         filehandler = open(objName, 'r')
#         thisStar = pickle.load(filehandler)
        
        #Plot
#         RVP.all_spec_overlap(thisStar, booSave = booSave,booShow = booShow )
        
        #Plot
#         RVP.RVs_single_star(thisStar, sigmaClip = -1, RVClip = 3000)

            
#         thisStar = None
#         print ''
# else:
#     print 'No red_*.obj files here.'
    
