#!/usr/bin/python

# import os
# import pickle
# import glob
# import pyfits as pf
import numpy as np
# import sys
# import RVPlots as RVP
# import pylab as plt
import RVTools as RVT


starIdx = 38

baryRVs = np.load('npy/baryRVs.npy')
MJDs = np.load('npy/MJDs.npy')
avgBaryRVs = np.load('npy/avgBaryRVs.npy')
avgMJDs = np.load('npy/avgMJDs.npy')
data = np.load('npy/data.npy')

RVT.fit_sine_RVs(baryRVs, MJDs, data, RVClip = 5e4, starIdx = starIdx)
RVT.fit_sine_RVs(avgBaryRVs, avgMJDs, data, RVClip = 5e4, starIdx = starIdx, npyName = 'avgSineFit.npy')


sineFit = np.load('npy/sineFit.npy')
avgSineFit = np.load('npy/avgSineFit.npy')
RVT.fit_results(data, sineFit, avgSineFit, baryRVs, avgBaryRVs)