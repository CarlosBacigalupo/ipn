# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import reduce_2dfdr 
import os
import glob

# <codecell>

#instantiate and set working folders
dr2df = reduce_2dfdr.dr2df()

#2dfdr program dir
dr2df.dr_dir = '/Users/Carlos/Documents/workspace/2dfdr/6.0/src_code/2dfdr-6.0/bin'
os.environ['PATH'] = '/Users/Carlos/Documents/workspace/2dfdr/6.0/src_code/2dfdr-6.0/bin:' + os.environ['PATH']

# <codecell>

# #RV precision run BIAS

# #target folder
# # batch_dir = '/Users/Carlos/Documents/HERMES/reductions/batch_tests/'
# batch_dir = '/Users/Carlos/Documents/HERMES/reductions/rhoTuc_long/bias/'

# #all reduced (*red.fits) files will be copied to this folder
# dr2df.final_dir = '/Users/Carlos/Documents/HERMES/reductions/rhoTuc_long/bias/'


# #the following arrays have n rows corresponding to n observations (can have more than 1 science file per observation)
# #array of folders with source files
# source_dir_array = ['/Users/Carlos/Documents/HERMES/data/140820/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140821/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140822/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140824/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140825/data/']

# #file prefixes to build file list for each observation
# date_array = ['20aug',
#               '21aug',
#               '22aug', 
#               '24aug', 
#               '25aug'] 

# #indices of each file set
# ix_array = [range(54,64),
#             range(1,11),
#             range(1,11),
#             range(1,11),
#             range(4,14)]

# <codecell>

# #RV run, HD1581 SCIENCE

# #target folder
# batch_dir = '/Users/Carlos/Documents/HERMES/reductions/HD1581_5.76/'

# #all reduced (*red.fits) files will be copied to this folder
# dr2df.final_dir = '/Users/Carlos/Documents/HERMES/reductions/HD1581_5.76/'


# #the following arrays have n rows corresponding to n observations (can have more than 1 science file per observation)
# #array of folders with source files
# source_dir_array = ['/Users/Carlos/Documents/HERMES/data/140820/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140821/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140822/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140824/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140825/data/']

# #file prefixes to build file list for each observation
# date_array = ['20aug',
#               '21aug',
#               '22aug',
#               '24aug',
#               '25aug'] 

# #indices of each file set with original arcs
# ix_array = [[34,52,53],
#             range(41,44)+[46,47],
#             [31,32]+range(36,39),
#             [53,54]+range(58,63),
#             [39,43]+range(44,47)]


# #index of the flats (within file set)
# flat_array=[0, -1, 1, 0, 0]

# #index of the arcs (within file set)
# arc_array=[1, -2, 0, 1, 1]
 

# #corresponing bias (looks for BIASCombined.fits in the folders bias/<this index>_*/<camera#>/
# bias_array=[0,1,2,3,4]

# <codecell>

# #RV run, 47Tuc SCIENCE

# #target folder
# batch_dir = '/Users/Carlos/Documents/HERMES/reductions/47Tuc_center/'

# #all reduced (*red.fits) files will be copied to this folder
# dr2df.final_dir = '/Users/Carlos/Documents/HERMES/reductions/47Tuc_center/'


# #the following arrays have n rows corresponding to n observations (can have more than 1 science file per observation)
# #array of folders with source files
# source_dir_array = ['/Users/Carlos/Documents/HERMES/data/140821/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140822/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140822/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140824/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140825/data/']

# #file prefixes to build file list for each observation
# date_array = ['21aug',
#               '22aug',
#               '22aug',
#               '24aug',
#               '25aug'] 

# #indices of each file set with original arcs
# ix_array = [range(25,31),
#             range(26,31),
#             range(48,53),
#             [48,49]+range(50,53),
#             [34,38]+range(35,38)]


# #index of the flats (within file set)
# flat_array=[0,-1,-2, 0, 0]

# #index of the arcs (within file set)
# arc_array=[1,0,-1, 1, 1]
 
# #corresponing bias (looks for BIASCombined.fits in the folders bias/<this index>_*/<camera#>/
# bias_array=[1,2,2,3,4]

# <codecell>

# #RV run, HD285507

# #target folder
# batch_dir = '/Users/Carlos/Documents/HERMES/reductions/HD285507_6.0/'

# #all reduced (*red.fits) files will be copied to this folder
# dr2df.final_dir = '/Users/Carlos/Documents/HERMES/reductions/HD285507_6.0/'


# #the following arrays have n rows corresponding to n observations (can have more than 1 science file per observation)
# #array of folders with source files
# source_dir_array = ['/Users/Carlos/Documents/HERMES/data/140820/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140821/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140822/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140824/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140825/data/']

# #file prefixes to build file list for each observation
# date_array = ['20aug',
#               '21aug',
#               '22aug',
#               '24aug',
#               '25aug'] 

# #indices of each file set with original arcs
# ix_array = [[37,38]+range(39,42),
#             [36,37]+range(38,41),
#             [40,39]+range(41,44),
#             [66,67]+range(63,66),
#             [47,48]+range(49,52)]


# #index of the flats (within file set)
# flat_array=[0,0,0,0,0]

# #index of the arcs (within file set)
# arc_array=[1,1,1,1,1]
 
# #corresponing bias (looks for BIASCombined.fits in the folders bias/<this index>_*/<camera#>/
# bias_array=[0,1,2,3,4]

# <codecell>

# #RV run, rhoTuc (long series) SCIENCE

# #target folder
# batch_dir = '/Users/Carlos/Documents/HERMES/reductions/rhoTuc_6.0/'

# #all reduced (*red.fits) files will be copied to this folder
# dr2df.final_dir = '/Users/Carlos/Documents/HERMES/reductions/rhoTuc_6.0/'


# #the following arrays have n rows corresponding to n observations (can have more than 1 science file per observation)
# #array of folders with source files
# source_dir_array = ['/Users/Carlos/Documents/HERMES/data/140820/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140821/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140821/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140822/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140822/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140824/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140825/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140825/data/']

# #file prefixes to build file list for each observation
# date_array = ['20aug',
#               '21aug',
#               '21aug', 
#               '22aug',
#               '22aug',
#               '24aug',
#               '25aug',
#               '25aug'] 

# #indices of each file set with original arcs
# ix_array = [range(42,52),
#             range(31,36),
#             [44,46,47],
#             range(31,36),
#             range(44,48),
#             range(53,58),
#             range(39,44),
#             range(52,55)]


# #index of the arcs (within file set)
# arc_array=[1,1,1,0,1,1,-1, 1]
 
# #index of the flats (within file set)
# flat_array=[0,0,2,1,0,0,0,0]

# #corresponing bias (looks for BIASCombined.fits in the folders bias/<this index>_*/<camera#>/
# bias_array=[0,1,1,2,2,3,4,4]

# <codecell>

# #M67,BIAS, all observations, LOCAL

# #target folder
# # batch_dir = '/Users/Carlos/Documents/HERMES/reductions/batch_tests/'
# batch_dir = '/Users/Carlos/Documents/HERMES/reductions/batch_reductions/m67_5.70/bias/'

# #all reduced (*red.fits) files will be copied to this folder
# dr2df.final_dir = '/Users/Carlos/Documents/HERMES/reductions/batch_reductions/m67_5.70/bias/'


# #the following arrays have n rows corresponding to n observations (can have more than 1 science file per observation)
# #array of folders with source files
# source_dir_array = ['/Users/Carlos/Documents/HERMES/data/131217/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140111/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140112/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140207/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140208/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140209/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140210/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140211/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140107/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140109/data/']

# #file prefixes to build file list for each observation
# date_array = ['17dec',
#               '11jan', 
#               '12jan', 
#               '07feb', 
#               '08feb',
#               '09feb',
#               '10feb',
#               '11feb',
#               '07jan', 
#               '09jan'] 

# #indices of each file set
# ix_array = [range(5,15),
#             range(1,11),
#             range(1,11),
#             range(1,6),
#             range(1,11),
#             range(1,11),
#             range(1,11),
#             range(1,11),
#             range(13,25),
#             range(1,11)]

# <codecell>

# #M67, all observations LOCAL

# #target folder
# # batch_dir = '/Users/Carlos/Documents/HERMES/reductions/batch_tests/'
# batch_dir = '/Users/Carlos/Documents/HERMES/reductions/batch_reductions/m67_5.70/'

# #all reduced (*red.fits) files will be copied to this folder
# dr2df.final_dir = '/Users/Carlos/Documents/HERMES/reductions/batch_reductions/m67_5.70/'


# #the following arrays have n rows corresponding to n observations (can have more than 1 science file per observation)
# #array of folders with source files
# source_dir_array = ['/Users/Carlos/Documents/HERMES/data/131217/data/',
#                   '/Users/Carlos/Documents/HERMES/data/131217/data/',
#                   '/Users/Carlos/Documents/HERMES/data/131218/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140111/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140111/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140111/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140112/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140207/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140207/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140208/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140208/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140209/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140209/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140210/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140211/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140211/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140107/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140107/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140107/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140107/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140109/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140109/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140109/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140109/data/',
#                   '/Users/Carlos/Documents/HERMES/data/140109/data/']

# #file prefixes to build file list for each observation
# date_array = ['17dec',
#               '17dec',
#               '18dec',
#               '11jan', 
#               '11jan', 
#               '11jan', 
#               '12jan', 
#               '07feb', 
#               '07feb', 
#               '08feb',
#               '08feb',
#               '09feb',
#               '09feb',
#               '10feb',
#               '11feb',
#               '11feb',
#               '07jan', 
#               '07jan', 
#               '07jan', 
#               '07jan', 
#               '09jan', 
#               '09jan', 
#               '09jan', 
#               '09jan', 
#               '09jan'] 

# #indices of each file set with original arcs
# # ix_array = [range(39,44),
# #             range(44,49),
# #             range(30,36),
# #             range(26,31),
# #             range(31,36),
# #             range(36,41),
# #             range(28,33),
# #             range(20,25),
# #             range(25,33),
# #             range(25,30),
# #             range(30,38),
# #             range(27,32),
# #             range(32,37),
# #             range(25,34),
# #             range(11,19),
# #             range(19,27),
# #             range(47,52),
# #             range(52,57),
# #             range(57,62),
# #             range(62,65),
# #             range(26,30),
# #             range(30,36),
# #             range(36,39),
# #             range(45,48),
# #             range(48,53)]

# #indices of each file set(one arc per night)
# ix_array = [range(39,44),
#             range(44,48)+[43],
#             range(30,36),
#             range(26,31),
#             [26]+range(32,36),
#             [26]+range(37,41),
#             range(28,33),
#             range(20,25),
#             [25,21]+range(27,33),
#             range(25,30),
#             [30,26]+range(32,38),
#             range(27,32),
#             [32,28]+range(34,37),
#             range(25,34),
#             range(11,19),
#             [19,12]+range(21,27),
#             range(47,52),
#             [47]+range(53,57),
#             [47]+range(58,62),
#             [47]+range(63,65),
#             range(26,30),
#             range(30,34)+[28,35],
#             [36,28,38],
#             [45,28,47],
#             range(48,51)+[28,52]]

# #index of the arcs (within file set)
# arc_array=[4,4,4,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,2,4,1,1,3]
 
# #index of the flats (within file set)
# flat_array=[3,3,5,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,3,5,2,2,4]

# #corresponing bias (looks for BIASCombined.fits in the folders bias/<this index>_*/<camera#>/
# bias_array=[0,0,0,1,1,1,2,3,3,4,4,5,5,6,7,7,8,8,8,8,9,9,9,9,9]

# <codecell>

# #M67, all observations remote

# #target folder
# # batch_dir = '/Users/Carlos/Documents/HERMES/reductions/batch_tests/'
# batch_dir = '/disks/ceres/makemake/aphot/kalumbe/reductions/m67_all/'

# #all reduced (*red.fits) files will be copied to this folder
# dr2df.final_dir = '/disks/ceres/makemake/aphot/kalumbe/reductions/m67_all/'


# #the following arrays have n rows corresponding to n observations (can have more than 1 science file per observation)
# #array of folders with source files
# source_dir_array = ['/disks/ceres/makemake/aphot/kalumbe/hermes/data/131217/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/131217/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/131218/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140111/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140111/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140111/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140112/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140207/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140207/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140208/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140208/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140209/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140209/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140210/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140211/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140211/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140107/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140107/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140107/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140107/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140109/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140109/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140109/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140109/data/',
#                   '/disks/ceres/makemake/aphot/kalumbe/hermes/data/140109/data/']

# #file prefixes to build file list for each observation
# date_array = ['17dec',
#               '17dec',
#               '18dec',
#               '11jan', 
#               '11jan', 
#               '11jan', 
#               '12jan', 
#               '07feb', 
#               '07feb', 
#               '08feb',
#               '08feb',
#               '09feb',
#               '09feb',
#               '10feb',
#               '11feb',
#               '11feb',
#               '07jan', 
#               '07jan', 
#               '07jan', 
#               '07jan', 
#               '09jan', 
#               '09jan', 
#               '09jan', 
#               '09jan', 
#               '09jan'] 

# #indices of each file set
# ix_array = [range(39,44),
#             range(44,49),
#             range(30,36),
#             range(26,31),
#             range(31,36),
#             range(36,41),
#             range(28,33),
#             range(20,25),
#             range(25,33),
#             range(25,30),
#             range(30,38),
#             range(27,32),
#             range(32,37),
#             range(25,34),
#             range(11,19),
#             range(19,27),
#             range(47,52),
#             range(52,57),
#             range(57,62),
#             range(62,65),
#             range(26,30),
#             range(30,36),
#             range(36,39),
#             range(45,48),
#             range(48,53)]

# #index of the arcs (within file set)
# arc_array=[4,4,4,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,2,4,1,1,3]
 
# #index of the flats (within file set)
# flat_array=[3,3,5,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,3,5,2,2,4]

# <codecell>

# #HR spec_PSF (There are more to add)

# #target folder
# batch_dir = '/Users/Carlos/Documents/HERMES/reductions/PSF_spec/'

# #all reduced (*red.fits) files will be copied to this folder
# dr2df.final_dir = '/Users/Carlos/Documents/HERMES/reductions/PSF_spec/'

# #the following arrays have n rows corresponding to n observations (can have more than 1 science file per observation)

# #array of folders with source files

# source_dir_array = ['/Users/Carlos/Documents/HERMES/data/140207/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140207/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140207/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140207/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140207/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140207/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140207/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140208/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140208/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140208/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140208/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140208/data/',
#                     '/Users/Carlos/Documents/HERMES/data/140208/data/']



# #file prefixes to build file list for each observation

# date_array = ['07feb',

#               '07feb',

#               '07feb',

#               '07feb',

#               '07feb',

#               '07feb',

#               '08feb',

#               '08feb',

#               '08feb',

#               '08feb',

#               '08feb',

#               '08feb'] 



# #indices of each file set

# ix_array = [range(12,20),

#             range(20,25),

#             range(25,33),
            
#             range(33,38),
            
#             range(41,46),
            
#             range(46,51),
            
#             range(51,57),
            
#             range(17,25),
            
#             range(25,30),
            
#             range(30,38),
            
#             range(38,44),
            
#             range(44,49),
            
#             range(49,54)]



# #index of the arcs (within file set)

# arc_array=[1,1,1,1,1,1,1,1,1,1,1,1,1]

 

# #index of the flats (within file set)

# flat_array=[0,0,0,0,0,0,0,0,0,0,0,0,0]

# <codecell>

# #BIAS
# #travels the arrays specified above running a full 2dfdr reduction in each loop

# for i in range(len(date_array)):
#     if i>3: #to skip a few filesets when needed
#         print date_array[i]
#         dr2df.source_dir = source_dir_array[i]
#         dr2df.target_dir = batch_dir + str(i) + '_'+ date_array[i] +'/'
#         dr2df.file_ix = ix_array[i]
#         dr2df.date = date_array[i] 
# #         dr2df.bias_reduce(copyFiles = True, idxFile = '5.70_0709_slm.idx') 
#         dr2df.bias_reduce(copyFiles = True, idxFile = '5.70slm.idx') #fake idx name to make 2dfdr fail and copy only

# <codecell>

# glob.glob(batch_dir + 'bias/'+str(bias_array[i])+'_*')[0] + '/'
print batch_dir + 'bias/'+str(bias_array[i])+'_*'

# <codecell>

#SCIENCE
#travels the arrays specified above running a full 2dfdr reduction in each loop

for i in range(len(date_array)):
    if i>1: #to skip a few filesets when needed, -1 to do all
        print date_array[i]
        dr2df.source_dir = source_dir_array[i]
        dr2df.target_dir = batch_dir + str(i) + '_'+ date_array[i] +'/'
        dr2df.file_ix = ix_array[i]
        dr2df.date = date_array[i] 
        dr2df.arc = arc_array[i] 
        dr2df.flat = flat_array[i] 
        dr2df.bias_dir = glob.glob(batch_dir + 'bias/'+str(bias_array[i])+'_*')[0] + '/'
        print dr2df.bias_dir
        dr2df.auto_reduce(copyFiles = False, idxFile = 'rhoTuc.idx', reduce = True) 

# <codecell>

#james allen code to combine

import subprocess
import os
import tempfile
import shutil

def run_2dfdr_combine(input_path_list, output_path, return_to=None, 
                      lockdir=LOCKDIR, **kwargs):
    """Run 2dfdr to combine the specified FITS files."""
    if len(input_path_list) < 2:
        raise ValueError('Need at least 2 files to combine!')
    output_dir, output_filename = os.path.split(output_path)
    # Need to extend the default timeout value; set to 5 hours here
    timeout = '300'
    # Write the 2dfdr AutoScript
    script = []
    for input_path in input_path_list:
        script.append('lappend glist ' +
                      os.path.relpath(input_path, output_dir))
    script.extend(['proc Quit {status} {',
                   '    global Auto',
                   '    set Auto(state) 0',
                   '}',
                   'set task DREXEC1',
                   'global Auto',
                   'set Auto(state) 1',
                   ('ExecCombine $task $glist ' + output_filename +
                    ' -success Quit')])
    script_filename = '2dfdr_script.tcl'
    with visit_dir(output_dir, return_to=return_to, lockdir=lockdir):
        # Print the script to file
        with open(script_filename, 'w') as f_script:
            f_script.write('\n'.join(script))
        # Run 2dfdr
        options = ['-AutoScript',
                   '-ScriptName',
                   script_filename,
                   '-Timeout',
                   timeout]
        run_2dfdr(output_dir, options, lockdir=None, **kwargs)
        # Clean up the script file
        os.remove(script_filename)
    return

# <codecell>


