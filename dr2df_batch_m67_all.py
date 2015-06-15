import HERMES
import os

#instantiate and set working folders
dr2df = HERMES.dr2df()

#2dfdr program dir
dr2df.dr_dir = '/home/science/staff/kalumbe/2dfdr/2dfdr_install/bin/'

#M67, all observations

#target folder
# batch_dir = '/Users/Carlos/Documents/HERMES/reductions/batch_tests/'
batch_dir = '/disks/ceres/makemake/aphot/kalumbe/reductions/m67_all/'

#all reduced (*red.fits) files will be copied to this folder
dr2df.final_dir = '/disks/ceres/makemake/aphot/kalumbe/reductions/m67_all/'

#the following arrays have n rows corresponding to n observations (can have more than 1 science file per observation)
#array of folders with source files
source_dir_array = ['/disks/ceres/makemake/aphot/kalumbe/hermes/131217/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/131217/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/131218/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140111/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140111/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140111/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140112/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140207/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140207/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140208/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140208/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140209/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140209/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140210/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140211/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140211/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140107/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140107/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140107/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140107/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140109/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140109/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140109/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140109/data/',
                  '/disks/ceres/makemake/aphot/kalumbe/hermes/140109/data/']

#file prefixes to build file list for each observation
date_array = ['17dec',
              '17dec',
              '18dec',
              '11jan', 
              '11jan', 
              '11jan', 
              '12jan', 
              '07feb', 
              '07feb', 
              '08feb',
              '08feb',
              '09feb',
              '09feb',
              '10feb',
              '11feb',
              '11feb',
              '07jan', 
              '07jan', 
              '07jan', 
              '07jan', 
              '09jan', 
              '09jan', 
              '09jan', 
              '09jan', 
              '09jan'] 

#indices of each file set
ix_array = [range(39,44),
            range(44,49),
            range(30,36),
            range(26,31),
            range(31,36),
            range(36,41),
            range(28,33),
            range(20,25),
            range(25,33),
            range(25,30),
            range(30,38),
            range(27,32),
            range(32,37),
            range(25,34),
            range(11,19),
            range(19,27),
            range(47,52),
            range(52,57),
            range(57,62),
            range(62,65),
            range(26,30),
            range(30,36),
            range(36,39),
            range(45,48),
            range(48,53)]

#index of the arcs (within file set)
arc_array=[4,4,4,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,2,4,1,1,3]
 
#index of the flats (within file set)
flat_array=[3,3,5,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,3,5,2,2,4]

#travels the arrays above running a full 2dfdr reduction in each loop
for i in range(len(date_array)):
    if i>-1: #to skip a few file sets when needed
        print date_array[i]
        dr2df.source_dir = source_dir_array[i]
        dr2df.target_dir = batch_dir + str(i) + '_'+ date_array[i] +'/'
        dr2df.file_ix = ix_array[i]
        dr2df.date = date_array[i] 
        dr2df.arc = arc_array[i] 
        dr2df.flat = flat_array[i] 
        dr2df.auto_reduce(copyFiles = True, idxFile = '5.69_0618_fpsf_skymed_slm.idx') 

