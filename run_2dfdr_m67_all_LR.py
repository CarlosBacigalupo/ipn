import reduce_2dfdr 
import os
import numpy as np


# M67, all observations


#reduction flags
useBias = False
copyFiles = True
doReduce = True
overwrite = True
idxFile = 'no_flat_no_bias.idx'
startFrom = 1 #number of data set to begin with. 0 for beginning. Good for starting half way through if it cancelled
reduceMode = 'one_arc'


#path to 2dfdr
dr_dir = '/home/staff/mq20101889/2dfdr/6.2/2dfdr_install/bin'

#target directory. It will copy the data files to sub-directories branching from this directory
target_root = '/home/staff/mq20101889/HERMES/reductions/test/'

#all science reduced (*red.fits) files will be copied to this directory
final_dir = '/home/staff/mq20101889/HERMES/reductions/test/'

#path to data source
galah_data_root = '/home/staff/mq20101889/galah_data/'


#len(date_list) = number of observations (can have more than 1 science file per observation)
date_list = ['131217',
            '131217',
            '131218',
            '140111',
            '140111',
            '140111',
            '140112',
            '140209',
            '140209',
            '140210',
            '140211',
            '140211',
            '140107',
            '140107',
            '140107',
            '140107',
            '140109',
            '140109',
            '140109',
            '140109',
            '140109']

date_list = ['140209',
            '140209',
            '140210',
            '140211',
            '140211',
            '140107',
            '140107',
            '140107',
            '140107',
            '140109',
            '140109',
            '140109',
            '140109',
            '140109']


#Filenumbers for each dataset FLAT_IDX, ARC_IDX, SCIENCE_IDX[S]. Asumes [flat, arc, sci, sci, ...]
ix_array = [[42,43]+range(39,42),
            [47,48]+range(44,47),
            [35,34]+range(30,34),
            [27,26]+range(28,31),
            [32,31]+range(33,36),
            [37,36]+range(38,41),
            [29,28]+range(30,33),
            [27,28]+range(29,32),
            [32,33]+range(34,37),
            [25,26]+range(27,34),
            [11,12]+range(13,19),
            [19,20]+range(21,27),
            [48,47]+range(49,52),
            [53,52]+range(54,57),
            [58,57]+range(59,62),
            [63,62]+[64],
            [29,28]+range(26,28),
            [35,34]+range(30,34),
            [38,37]+[36],
            [47,46]+[45],
            [52,51]+range(48,51)]

ix_array = [[27,28]+range(29,32),
            [32,33]+range(34,37),
            [25,26]+range(27,34),
            [11,12]+range(13,19),
            [19,20]+range(21,27),
            [48,47]+range(49,52),
            [53,52]+range(54,57),
            [58,57]+range(59,62),
            [63,62]+[64],
            [29,28]+range(26,28),
            [35,34]+range(30,34),
            [38,37]+[36],
            [47,46]+[45],
            [52,51]+range(48,51)]

#adds 2dfdr bi path to PATH variable
os.environ['PATH'] = dr_dir + ':' + os.environ['PATH'] 

#compose absolute path names
source_dir_array = [galah_data_root + s + '/data/' for s in date_list]

#compose file prefixes from date_list
months = np.array(['', 'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
d = np.array([s[4:] for s in date_list])
m = months[np.array([s[2:4] for s in date_list]).astype(int)]
filename_prfx = np.core.defchararray.add(d, m)

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
