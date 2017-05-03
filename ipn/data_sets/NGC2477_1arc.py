#RV run, NGC2477 1arc

reduceMode = 'one_arc'

#target directory. It will copy the data files to sub-directories branching from this directory
target_root = 'NGC2477_1arc/' 

#len(date_list) = number of observations (can have more than 1 science file per observation)
date_list = ['140107',
             '140107',
             '140107',
             '140109',
             '140109',
             '140109',
             '140109',
             '140109']

root_date_link = [0,0,0,0,0,0,0,0]

#Filenumbers for each dataset FLAT_IDX, ARC_IDX, SCIENCE_IDX[S]. Assumes [flat, arc, sci, sci, ...]
ix_array = [[27,25]+range(28,31),
            [31,32]+range(33,37),
            [38,37]+range(39,47),
            [20,19]+range(16,19),
            [25,24]+range(21,24),
            [44,43]+range(39,43),
            [55,54,53],
            [60,59]+range(56,59)]
  