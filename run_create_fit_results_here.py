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


baryRVs = np.load('npy/baryRVs.npy')
avgBaryRVs = np.load('npy/avgBaryRVs.npy')
data = np.load('npy/data.npy')
sineFit = np.load('npy/sineFit.npy')
avgSineFit = np.load('npy/avgSineFit.npy')

RVT.fit_results(data, sineFit, avgSineFit, baryRVs, avgBaryRVs)