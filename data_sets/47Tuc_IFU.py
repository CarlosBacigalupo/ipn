#47Tuc core 

reduceMode = 'starting_set'

#target directory. It will copy the data files to sub-directories branching from this directory
target_root = '47Tuc_IFU/'

#len(date_list) = number of observations (can have more than 1 science file per observation)
date_list = ['140820']

root_date_link = [2]

#Filenumbers for each dataset FLAT_IDX, ARC_IDX, SCIENCE_IDX[S]. Assumes [flat, arc, sci, sci, ...]
ix_array = [[18,19,20]]

