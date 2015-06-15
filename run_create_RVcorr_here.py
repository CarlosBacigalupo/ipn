#!/opt/local/bin/python

# import os
import numpy as np
import RVTools as RVT
# import sys
# import toolbox

RVClip = 2000
starSet = []

data=np.load('npy/data.npy')
RVs=np.load('npy/RVs.npy')
SNRs=np.load('npy/SNRs.npy')
# starSet=np.load('npy/starSet.npy')


allW = RVT.create_allW(data, SNRs, starSet = starSet, RVCorrMethod = 'PM')
RVCorr = RVT.create_RVCorr(RVs, allW, RVClip = RVClip, starSet = starSet)
np.save('npy/allW_PM.npy', allW)
np.save('npy/RVCorr_PM.npy', RVCorr)

allW = RVT.create_allW(data, SNRs, starSet = starSet, RVCorrMethod = 'DM')
RVCorr = RVT.create_RVCorr(RVs, allW, RVClip = RVClip, starSet = starSet)
np.save('npy/allW_DM.npy', allW)
np.save('npy/RVCorr_DM.npy', RVCorr)



