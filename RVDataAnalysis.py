# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pandas as pd

# <codecell>

cd ~/Documents/HERMES/reductions/47Tuc_core_6.2/

# <codecell>

cd ~/Documents/HERMES/reductions/47Tuc_core_1arc_6.2/

# <codecell>

cd ~/Documents/HERMES/reductions/HD285507_6.2/

# <codecell>

cd ~/Documents/HERMES/reductions/HD285507_1arc_6.2/

# <codecell>

cd ~/Documents/HERMES/reductions/m67_6.2/

# <codecell>

cd ~/Documents/HERMES/reductions/m67_1arc_6.2/

# <codecell>

cd ~/Documents/HERMES/reductions/NGC2477_6.2/

# <codecell>

cd ~/Documents/HERMES/reductions/NGC2477_1arc_6.2/

# <codecell>

data=np.load('npy/data.npy')
RVs=np.load('npy/RVs.npy')
sigmas=np.load('npy/sigmas.npy')
baryVels=np.load('npy/baryVels.npy')
JDs=np.load('npy/JDs.npy')

# <codecell>

thisSlice = RVs
totalStars = RVs.shape[0]

# <codecell>

goodStars = np.sum(((thisSlice<5000) & (thisSlice!=0)), axis=0).astype(float)

# <codecell>

size = np.array(thisSlice.shape)
total = np.reshape(np.repeat(size[0], size[1]*size[2]),(size[1],size[2]))
labels = ['Blue','Green','Red','IR']

# <codecell>

a = pd.DataFrame(goodStars/total*100)
a.columns = labels

# <codecell>

a.to_csv('aaa.csv')

# <codecell>

a

# <codecell>


# <codecell>

# np.set_printoptions(precision=3)
goodStars/total*100

# <codecell>

goodStars

# <codecell>


