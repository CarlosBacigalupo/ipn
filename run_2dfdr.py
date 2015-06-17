#!/usr/bin/python

import reduce_2dfdr 
import os
import numpy as np
import sys
import time
import importlib



#reduction flags
ver = 6.5
booLog = False
useBias = False
copyFiles = True
doReduce = True
overwrite = False
idxFile = 'no_flat_no_bias.idx'
startFrom = 4 #number of data set to begin with. 0 for beginning. Good for starting half way through if it cancelled


# Any observations
reduceSet = -1
reduceCam = -1
final_dir =''

if len(sys.argv)>1:
    dataset = sys.argv[1]
    try:
        thisDataset = importlib.import_module('data_sets.'+dataset)
    except:
        print 'Could not load',dataset         
        sys.exit()
    
    reduceMode = thisDataset.reduceMode         
        
    if len(sys.argv)>2:
        reduceSet = int(sys.argv[2])
        reduceMode = 'single_set'
        if len(sys.argv)>3:
            reduceCam = int(sys.argv[3])-1
        
    
    location = 'mylaptop'
    if os.path.expanduser('~')=='/home/staff/mq20101889':
        location = 'nut'
    
    if location=='mylaptop':
        #target directory. It will copy the data files to sub-directories branching from this directory
        target_root = '/Users/Carlos/Documents/HERMES/reductions/'+str(ver)+'/'+thisDataset.target_root #my laptop

        #path to 2dfdr
        if ver==6.5: dr_dir = '/Users/Carlos/Documents/workspace/2dfdr/6.5/binaries-macosx-Lion/2dfdr_install/bin' #my laptop 6.5
            
    elif location=='nut':
        
        #target directory. It will copy the data files to sub-directories branching from this directory
        target_root = '/disks/nut/data/mq20101889/HERMES/reductions/'+str(ver)+'/'+thisDataset.target_root 

        #path to 2dfdr
        if ver==6.5: dr_dir='/disks/nut/data/mq20101889/2dfdr/6.5/binaries-linux/2dfdr_install/bin' #in nut 6.5            
        if ver==6.4: dr_dir='/home/staff/mq20101889/2dfdr/6.4/2dfdr_install/bin' #in nut 6.4
       
    print location, 'v',ver

    if final_dir =='':
        final_dir = target_root
        
    if copyFiles==True:
        try:
            os.mkdir(target_root)
        except:
            pass



    #path to data sources
    HERMES_data_root = []
    if location=='mylaptop': 
        HERMES_data_root.append('')
        HERMES_data_root.append('')
        HERMES_data_root.append('/Users/Carlos/Documents/HERMES/data/')
        HERMES_data_root.append('')
    elif location=='nut':
        HERMES_data_root.append('/home/staff/mq20101889/galah_data/')
        HERMES_data_root.append('/home/staff/mq20101889/galah_pilot/')
        HERMES_data_root.append('/home/staff/mq20101889/RV_data/')
        HERMES_data_root.append('/home/staff/mq20101889/jan14_data/')
    HERMES_data_root = np.array(HERMES_data_root)
    
    #compose absolute path names
    source_dir_array = np.core.defchararray.add(HERMES_data_root[np.array(thisDataset.root_date_link)], [s + '/data/' for s in thisDataset.date_list]) 
    
    #compose file prefixes from date_list
    months = np.array(['', 'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
    d = np.array([s[4:] for s in thisDataset.date_list])
    m = months[np.array([s[2:4] for s in thisDataset.date_list]).astype(int)]
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
    dr2df.ix_array = thisDataset.ix_array
    dr2df.filename_prfx = filename_prfx
    dr2df.date_list = thisDataset.date_list
    dr2df.source_dir_array = source_dir_array
         
    #run forest, run
    if booLog==True: sys.stdout = open(str(startFrom)+str(reduceSet)+'_'+str(time.strftime('%X'))+'.log', 'w')
                      
    print time.strftime('%X %x %Z'), '  Starting reduction'
    dr2df.runReduction()
#     dr2df.create_folders()
#     dr2df.collect_red_files()
    print time.strftime('%X %x %Z'), '  Ending reduction'
    
    # n=0
    # for dataset, fileN in zip(filename_prfx, ix_array):
    #     for i in fileN[2:]:
    #         print str(n)+'_'+str(dataset),'\t',str(i)
    #     n+=1

else:
    print 'No data_set specified.'
    print 'run_2dfdr.py data_set [reduceSet] [reduceCam]'
    
    
    