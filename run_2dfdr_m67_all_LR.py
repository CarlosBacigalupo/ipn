# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import reduce_2dfdr 
reload(reduce_2dfdr)
import os
import numpy as np

# <headingcell level=6>

# M67, all observations

# <codecell>

#reduction flags
useBias = False
copyFiles = True
doReduce = True
overwrite = True
idxFile = 'no_flat_no_bias.idx'
startFrom = 1 #number of data set to begin with. 0 for beginning. Good for starting half way through if it cancelled
reduceMode = 'one_arc'

# <codecell>

#meta_data full

#path to 2dfdr
dr_dir = '/Users/Carlos/Documents/workspace/2dfdr/6.0/src_code/2dfdr-6.0/bin'

#target directory. It will copy the data files to sub-directories branching from this directory
target_root = '/disks/ceres/makemake/aphot/kalumbe/reductions/m67_all/'
target_root = '/Users/Carlos/Documents/HERMES/reductions/test/'

#all science reduced (*red.fits) files will be copied to this directory
final_dir = '/disks/ceres/makemake/aphot/kalumbe/reductions/m67_all/'
final_dir = '/Users/Carlos/Documents/HERMES/reductions/test/'

#path to data source
galah_data_root = '/disks/ceres/makemake/aphot/kalumbe/hermes/data/'
galah_data_root = '/Users/Carlos/Documents/HERMES/data/'

#len(date_list) = number of observations (can have more than 1 science file per observation)
date_list = ['140821',
            '140822',
            '140822',
            '140824',
            '140825']


#Filenumbers for each dataset FLAT_IDX, ARC_IDX, SCIENCE_IDX[S]. Asumes [flat, arc, sci, sci, ...]
ix_array = [[25,26]+range(39,42),
            [30,26]+range(44,47),
            [51,52]+range(30,34),
            [48,49]+range(50,53),
            [34,38]+range(35,38)]

# <codecell>

# #meta_data full

# #path to 2dfdr
# dr_dir = '/Users/Carlos/Documents/workspace/2dfdr/6.0/src_code/2dfdr-6.0/bin'

# #target directory. It will copy the data files to sub-directories branching from this directory
# target_root = '/disks/ceres/makemake/aphot/kalumbe/reductions/m67_all/'
# target_root = '/Users/Carlos/Documents/HERMES/reductions/test/'

# #all science reduced (*red.fits) files will be copied to this directory
# final_dir = '/disks/ceres/makemake/aphot/kalumbe/reductions/m67_all/'
# final_dir = '/Users/Carlos/Documents/HERMES/reductions/test/'

# #path to data source
# galah_data_root = '/disks/ceres/makemake/aphot/kalumbe/hermes/data/'
# galah_data_root = '/Users/Carlos/Documents/HERMES/data/'


# #len(date_list) = number of observations (can have more than 1 science file per observation)
# date_list = ['131217',
#             '131217',
#             '131218',
#             '140111',
#             '140111',
#             '140111',
#             '140112',
#             '140209',
#             '140209',
#             '140210',
#             '140211',
#             '140211',
#             '140107',
#             '140107',
#             '140107',
#             '140107',
#             '140109',
#             '140109',
#             '140109',
#             '140109',
#             '140109']


# #Filenumbers for each dataset FLAT_IDX, ARC_IDX, SCIENCE_IDX[S]. Asumes [flat, arc, sci, sci, ...]
# ix_array = [[42,43]+range(39,42),
#             [47,48]+range(44,47),
#             [35,34]+range(30,34),
#             [27,26]+range(28,31),
#             [32,31]+range(33,36),
#             [37,36]+range(38,41),
#             [29,28]+range(30,33),
#             [27,28]+range(29,32),
#             [32,33]+range(34,37),
#             [25,26]+range(27,34),
#             [11,12]+range(13,19),
#             [19,20]+range(21,27),
#             [48,47]+range(49,52),
#             [53,52]+range(54,57),
#             [58,57]+range(59,62),
#             [63,62]+[64],
#             [29,28]+range(26,28),
#             [35,34]+range(30,34),
#             [38,37]+[36],
#             [47,46]+[45],
#             [52,51]+range(48,51)]

# <codecell>

#adds 2dfdr bi path to PATH variable
os.environ['PATH'] = dr_dir + ':' + os.environ['PATH'] 

#compose absolute path names
source_dir_array = [galah_data_root + s + '/data/' for s in date_list]


#compose file prefixes from date_list
months = np.array(['', 'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
d = np.array([s[4:] for s in date_list])
m = months[np.array([s[2:4] for s in date_list]).astype(int)]
filename_prfx = np.core.defchararray.add(d, m)

# <codecell>

#instantiate class
dr2df = reduce_2dfdr.dr2df()

#pass variables to class
dr2df.dr_dir = dr_dir
dr2df.final_dir = final_dir
dr2df.overwrite = overwrite
dr2df.target_root = target_root
dr2df.galah_data_root = galah_data_root
dr2df.reduceMode = reduceMode
dr2df.idxFile = idxFile
dr2df.copyFiles = copyFiles
dr2df.doReduce = doReduce

#arrays
dr2df.ix_array = ix_array
dr2df.filename_prfx = filename_prfx
dr2df.date_list = date_list
dr2df.source_dir_array = source_dir_array
     
#run forest, run
dr2df.runReduction()

# <codecell>

dr2df.filename_prfx

# <codecell>

import shutil

# <codecell>

shutil.copyfile('/Users/Carlos/Documents/a.txt', '/Users/Carlos/Documents/c')

# <codecell>


