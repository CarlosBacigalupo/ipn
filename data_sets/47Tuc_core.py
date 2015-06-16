
#target directory. It will copy the data files to sub-directories branching from this directory
target_root = '47Tuc_core/' #my laptop

#len(date_list) = number of observations (can have more than 1 science file per observation)
date_list = ['140821',
             '140822',
             '140822',
             '140824',
             '140825']

root_date_link = np.array([2,2,2,2,2])

#Filenumbers for each dataset FLAT_IDX, ARC_IDX, SCIENCE_IDX[S]. Assumes [flat, arc, sci, sci, ...]
ix_array = [range(25,31),
            [30,26]+range(27,30),
            [51,52]+range(48,51),
            [48,49]+range(50,53),
            [34,38]+range(35,38)]
