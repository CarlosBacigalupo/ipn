#RV run, rhoTuc

reduceMode = 'starting_set'

#target directory. It will copy the data files to sub-directories branching from this directory
target_root = 'rhoTuc/'

#len(date_list) = number of observations (can have more than 1 science file per observation)
date_list = ['140820',
             '140821',
             '140821',
             '140822',
             '140822',
             '140824',
             '140825',
             '140825']

root_date_link = [2,2,2,2,2,2,2,2]


#Filenumbers for each dataset FLAT_IDX, ARC_IDX, SCIENCE_IDX[S]. Assumes [flat, arc, sci, sci, ...]
ix_array = [range(42,52),
            range(31,36),
            [47,46]+[44],
            [32,31]+range(33,36),
            range(44,48),
            range(53,58),
            [39,43]+range(40,43),
            [53,54,52]]
