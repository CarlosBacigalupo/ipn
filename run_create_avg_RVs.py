#!/usr/bin/python


import numpy as np
import RVTools as RVT

MJDs = np.load('npy/MJDs.npy')

RVT.avg_MJDs_groups(MJDs)
RVT.avg_MJDs_data()
