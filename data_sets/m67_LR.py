
#target directory. It will copy the data files to sub-directories branching from this directory
target_root = '/m67_lr/' 

#len(date_list) = number of observations (can have more than 1 science file per observation)
date_list = ['131217',
            '131217',
            '131218',
            '140111',
            '140111',
            '140111',
            '140112',
            '140209',
            '140209',
            '140210',
            '140211',
            '140211',
            '140107',
            '140107',
            '140107',
            '140107',
            '140109',
            '140109',
            '140109',
            '140109',
            '140109']

root_date_link = [1,1,1,1,1,1,1,0,0,0,0,0,3,3,3,3,3,3,3,3,3]

#Filenumbers for each dataset FLAT_IDX, ARC_IDX, SCIENCE_IDX[S]. Asumes [flat, arc, sci, sci, ...]
ix_array = [[42,43]+range(39,42),
            [47,48]+range(44,47),
            [35,34]+range(30,34),
            [27,26]+range(28,31),
            [32,31]+range(33,36),
            [37,36]+range(38,41),
            [29,28]+range(30,33),
            [27,28]+range(29,32),
            [32,33]+range(34,37),
            [25,26]+range(27,34),
            [11,12]+range(13,19),
            [19,20]+range(21,27),
            [48,47]+range(49,52),
            [53,52]+range(54,57),
            [58,57]+range(59,62),
            [63,62]+[64],
            [29,28]+range(26,28),
            [35,34]+range(30,34),
            [38,37]+[36],
            [47,46]+[45],
            [52,51]+range(48,51)]

reduceMode = 'starting_set'
