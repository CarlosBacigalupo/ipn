
reduceMode = 'starting_set'

#target directory. It will copy the data files to sub-directories branching from this directory
target_root = 'HD1581/' 

#len(date_list) = number of observations (can have more than 1 science file per observation)
date_list = ['140820',
             '140821',
             '140822',
             '140824',
             '140825']

root_date_link = [2,2,2,2,2]

#Filenumbers for each dataset FLAT_IDX, ARC_IDX, SCIENCE_IDX[S]. Assumes [flat, arc, sci, sci, ...]
ix_array = [[34,52,53],
            [47,46]+range(41,44),
            [32,31]+range(36,39),
            [53,54]+range(58,63),
            [39,43]+range(44,47)]
 
