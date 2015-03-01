import reduce_2dfdr 
import os
import numpy as np
import sys
import time


reduceCam = -1

# M67, all observations

reduceMode = 'one_arc'

reduceSet = -1
if len(sys.argv)>1:
    reduceSet = int(sys.argv[1])
    reduceMode = 'single_set'
    if len(sys.argv)>2:
        reduceCam = int(sys.argv[2])
        
    
    
#reduction flags
useBias = False
copyFiles = True
doReduce = True
overwrite = True
idxFile = 'no_flat_no_bias.idx'
startFrom = 0 #number of data set to begin with. 0 for beginning. Good for starting half way through if it cancelled


#path to 2dfdr
dr_dir = '/home/staff/mq20101889/2dfdr/6.2/2dfdr_install/bin' #in nut
# dr_dir = '/Users/Carlos/Documents/workspace/2dfdr/6.0/src_code/2dfdr-6.0/bin/' #local

#target directory. It will copy the data files to sub-directories branching from this directory
target_root = '/home/staff/mq20101889/HERMES/reductions/m67_lr_1arc_6.2/' #in nut
# target_root = '/home/staff/mq20101889/HERMES/reductions/test/' #in nut

#all science reduced (*red.fits) files will be copied to this directory
final_dir = '/home/staff/mq20101889/HERMES/reductions/m67_lr_1arc_6.2/'

#path to data sources
HERMES_data_root = []
HERMES_data_root.append('/home/staff/mq20101889/galah_data/')
HERMES_data_root.append('/home/staff/mq20101889/galah_pilot/')
HERMES_data_root.append('/home/staff/mq20101889/RV_data/')
HERMES_data_root.append('/home/staff/mq20101889/jan14_data/')
HERMES_data_root = np.array(HERMES_data_root)

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

root_date_link = np.array([1,1,1,1,1,1,1,0,0,0,0,0,3,3,3,3,3,3,3,3,3])

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


#adds 2dfdr bi path to PATH variable
os.environ['PATH'] = dr_dir + ':' + os.environ['PATH'] 

#compose absolute path names
source_dir_array = np.core.defchararray.add(HERMES_data_root[root_date_link], [s + '/data/' for s in date_list]) 

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
dr2df.reduceMode = reduceMode
dr2df.reduceSet = reduceSet
dr2df.reduceCam = reduceCam
dr2df.idxFile = idxFile
dr2df.copyFiles = copyFiles
dr2df.doReduce = doReduce
dr2df.startFrom = startFrom

#arrays
dr2df.ix_array = ix_array
dr2df.filename_prfx = filename_prfx
dr2df.date_list = date_list
dr2df.source_dir_array = source_dir_array
     
#run forest, run

sys.stdout = open(str(reduceSet)+'_'+str(time.strftime('%T'))+'.log', 'w')
                  
print time.strftime('%X %x %Z'), '  Starting reduction'
dr2df.runReduction()
print time.strftime('%X %x %Z'), '  Ending reduction'