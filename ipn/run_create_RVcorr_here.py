#!/opt/local/bin/python

# import os
import numpy as np
import RVTools as RVT
# import sys
# import toolbox

RVClip = 2000
starSet = []
refEpoch = 0

data=np.load('npy/data.npy')
RVs=np.load('npy/RVs.npy')
SNRs=np.load('npy/SNRs.npy')
# starSet=np.load('npy/starSet.npy')


allW_PM = RVT.create_allW(data, SNRs, starSet = starSet, RVCorrMethod = 'PM', refEpoch = refEpoch)
np.save('npy/allW_PM.npy', allW_PM)
allW_DM = RVT.create_allW(data, SNRs, starSet = starSet, RVCorrMethod = 'DM', refEpoch = refEpoch)
np.save('npy/allW_DM.npy', allW_DM)
 
RVCorr_PM = RVT.create_RVCorr_PM(RVs, allW_PM, RVClip = RVClip, starSet = starSet)
np.save('npy/RVCorr_PM.npy', RVCorr_PM)
RVCorr_DM = RVT.create_RVCorr_DM(RVs, allW_DM, RVClip = RVClip, starSet = starSet)
np.save('npy/RVCorr_DM.npy', RVCorr_DM)
 
cRVs_PM = RVs - RVCorr_PM
np.save('npy/cRVs_PM.npy', cRVs_PM)
cRVs_DM = RVs - RVCorr_DM
np.save('npy/cRVs_DM.npy', cRVs_DM)
cRVs_PMDM = RVs - RVCorr_PM - RVCorr_DM
np.save('npy/cRVs_PMDM.npy', cRVs_PMDM)
