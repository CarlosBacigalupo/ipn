# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from astropy.io import ascii
import numpy as np
import pandas as pd
from astropy import table

# <headingcell level=3>

# Create unique alias table

# <codecell>

cd ~/Documents/HERMES/reductions/6.5/HD1581/

# <codecell>

starAlias = np.load('starAlias.npy')

# <codecell>

starAlias = np.core.defchararray.replace(starAlias,'#','')

# <codecell>

starAliasUnique = []

maxLen = 0 
for i,thisStar in enumerate(np.unique(starAlias[:,0])):
    
    x = starAlias[np.where(thisStar==starAlias)[0],1]
    thisStarAliases = ", ".join(x)
    starAliasUnique.append([thisStar,thisStarAliases])
    
    #Get max len of alias column
    if len(thisStarAliases)>maxLen: maxLen =len(thisStarAliases)


print maxLen, 'is the length of the longest column'
starAliasUnique = np.array(starAliasUnique)

# <codecell>

labels = ['Name','Alias']
a = pd.DataFrame(starAliasUnique, dtype=str)
a.columns = labels
pd.set_printoptions(max_colwidth=maxLen)
print a.to_latex(index=False)

