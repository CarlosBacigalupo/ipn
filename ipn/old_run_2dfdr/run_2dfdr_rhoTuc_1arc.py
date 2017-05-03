#!/opt/local/bin/python

import reduce_2dfdr 
import os
import numpy as np
import sys
import time


reduceCam = -1

#RV run, rhoTuc

reduceMode = 'starting_set'

reduceSet = -1
if len(sys.argv)>1:
    reduceSet = int(sys.argv[1])
    reduceMode = 'single_set'
    if len(sys.argv)>2:
        reduceCam = int(sys.argv[2])
        
    
    
#reduction flags
useBias = False
copyFiles = False
doReduce = True
overwrite = False
idxFile = 'no_flat_no_bias.idx'
startFrom = 5 #number of data set to begin with. 0 for beginning. Good for starting half way through if it cancelled

#path to 2dfdr
# dr_dir = '/home/staff/mq20101889/2dfdr/6.2/2dfdr_install/bin' #in nut
dr_dir = '/Users/Carlos/Documents/workspace/2dfdr/6.2/2dfdr_install/bin/' #my laptop

#target directory. It will copy the data files to sub-directories branching from this directory
target_root = '/Users/Carlos/Documents/HERMES/reductions/6.2/rhoTuc_1arc_6.2/' #my laptop

#all science reduced (*red.fits) files will be copied to this directory
final_dir = '/Users/Carlos/Documents/HERMES/reductions/6.2/rhoTuc_1arc_6.2/'

#path to data sources
HERMES_data_root = []
HERMES_data_root.append('/Users/Carlos/Documents/HERMES/data/')
HERMES_data_root = np.array(HERMES_data_root)

#len(date_list) = number of observations (can have more than 1 science file per observation)
date_list = ['140820',
             '140821',
             '140821',
             '140822',
             '140822',
             '140824',
             '140825',
             '140825']

root_date_link = np.array([0,0,0,0,0,0,0,0])


#Filenumbers for each dataset FLAT_IDX, ARC_IDX, SCIENCE_IDX[S]. Assumes [flat, arc, sci, sci, ...]
ix_array = [range(42,52),
            range(31,36),
            [47,46]+[44],
            [32,31]+range(33,36),
            range(44,48),
            range(53,58),
            [39,43]+range(40,43),
            [53,54,52]]
#####End of custom data#################################################################



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

# sys.stdout = open(str(startFrom)+str(reduceSet)+'_'+str(time.strftime('%X'))+'.log', 'w')
                  
print time.strftime('%X %x %Z'), '  Starting reduction'
dr2df.runReduction()
print time.strftime('%X %x %Z'), '  Ending reduction'

#prints out the science file list to export to excel
# n=0
# for dataset, fileN in zip(filename_prfx, ix_array):
#     for i in fileN[2:]:
#         print str(n)+'_'+str(dataset),'\t',str(i)
#     n+=1
