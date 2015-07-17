#RV run, HD285507 1arc

reduceMode = 'one_arc'

#target directory. It will copy the data files to sub-directories branching from this directory
target_root = 'HD285507_1arc/'

#len(date_list) = number of observations (can have more than 1 science file per observation)
date_list = ['140820',
             '140821',
             '140822',
             '140824',
             '140825']

root_date_link = [2,2,2,2,2]


#Filenumbers for each dataset FLAT_IDX, ARC_IDX, SCIENCE_IDX[S]. Assumes [flat, arc, sci, sci, ...]
ix_array = [[37,38]+range(39,42),
            [36,37]+range(38,41),
            [40,39]+range(41,44),
            [66,67]+range(63,66),
            [47,48]+range(49,52)]
