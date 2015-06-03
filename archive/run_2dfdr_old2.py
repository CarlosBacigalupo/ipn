# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import reduce_2dfdr 
reload(reduce_2dfdr)
import os
# import glob
#instantiate class
dr2df = reduce_2dfdr.dr2df()

#2dfdr program dir
dr2df.dr_dir = '/Users/Carlos/Documents/workspace/2dfdr/6.0/src_code/2dfdr-6.0/bin'
os.environ['PATH'] = '/Users/Carlos/Documents/workspace/2dfdr/6.0/src_code/2dfdr-6.0/bin:' + os.environ['PATH']

#file with reduction parameters 
meta_file = 'meta_m67_all_LR'

#run forest, run
dr2df.doReduce(meta_file)

# <codecell>

#Reduce single arc style
#travels the arrays specified above running a full 2dfdr reduction in each loop
# dr2df.final_dir = final_dir
# importFlatArc = 0 #This forces all datasets to use the first(0th) reduced arc as the wl model

#first normal reduce the first dataset
# for i in range(len(date_array)):
#     if i>=startFrom: #to skip a few filesets when needed


# for i in range(len(date_array)):
#     if i>=startFrom: #to skip a few filesets when needed
#         print date_array[i]
#         dr2df.source_dir = source_dir_array[i]
#         dr2df.target_dir = batch_dir + str(i) + '_'+ date_array[i] +'/'
#         dr2df.file_ix = ix_array[i]
#         dr2df.date = date_array[i] 
#         dr2df.arc = arc_array[i] 
#         dr2df.flat = flat_array[i] 
# #         dr2df.bias_dir = glob.glob(batch_dir + 'bias/'+str(bias_array[i])+'_*')[0] + '/'
#         dr2df.auto_reduce(copyFiles = copyFiles, idxFile = idxFile, doReduce = doReduce) 


# <codecell>

# #SCIENCE
# #travels the arrays specified above running a full 2dfdr reduction in each loop
# dr2df.final_dir = final_dir

# for i in range(len(date_array)):
#     if i>=startFrom: #to skip a few filesets when needed
#         print date_array[i]
#         dr2df.source_dir = source_dir_array[i]
#         dr2df.target_dir = batch_dir + str(i) + '_'+ date_array[i] +'/'
#         dr2df.file_ix = ix_array[i]
#         dr2df.date = date_array[i] 
#         dr2df.arc = arc_array[i] 
#         dr2df.flat = flat_array[i] 
# #         dr2df.bias_dir = glob.glob(batch_dir + 'bias/'+str(bias_array[i])+'_*')[0] + '/'
#         dr2df.auto_reduce(copyFiles = copyFiles, idxFile = idxFile, doReduce = doReduce) 

