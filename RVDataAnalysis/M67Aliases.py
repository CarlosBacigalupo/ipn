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

cd ~/Documents/HERMES/reductions/6.5/m67_lr/

# <codecell>

starAlias = np.load('starAlias.npy')

# <codecell>

starAlias = np.core.defchararray.replace(starAlias,'Cl* NGC 2682','')
starAlias = np.core.defchararray.replace(starAlias,' ','')
starAlias = np.core.defchararray.replace(starAlias,'  ','')
starAlias = np.core.defchararray.replace(starAlias,'   ','')
starAlias = np.core.defchararray.replace(starAlias,'#','')

# <codecell>

starAlias = np.core.defchararray.strip(starAlias)

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

#print to latex
labels = ['Name','Alias']
a = pd.DataFrame(starAliasUnique, dtype=str)
a.columns = labels
pd.set_printoptions(max_colwidth=maxLen)
print a.to_latex(index=False)

# <headingcell level=3>

# Cross reference with M67 geller et al. table

# <codecell>

geller = ascii.read('aj518354t2_mrt.txt')

# <codecell>

def read_myStar(starAlias, myIdx):
    '''Parse star name, catalogue name and catalogue ID from starAlias 
    
    Parameters
    ----
    starAlias : np.array [obj name, alias]
        Array with the full list of aliases for each star
        
    myIdx : int
        index of starAlias array to parse

    '''
    catalogueDict = {'SAND': 'S',
                     'MMJ': 'M',
                     'FBC': 'F'}
    
    myStar = starAlias[myIdx,0]
#     print starAlias[myIdx,1]
    
    if starAlias[myIdx,1][:4]=='SAND':
        myCatalogue = 'S'
        myNumber = int(starAlias[myIdx,1][4:])
        
    elif ((starAlias[myIdx,1][:3]=='MMJ') and (starAlias[myIdx,1][3]!='S')):
        myCatalogue = 'M'
        myNumber = int(starAlias[myIdx,1][3:])
        
    elif starAlias[myIdx,1][:3]=='FBC':
        myCatalogue = 'F'
        myNumber = int(starAlias[myIdx,1][3:])
    
    else:
        myCatalogue, myNumber = 'O', 0
        
    return myStar, myCatalogue, myNumber

# <codecell>

def find_in_my_stars(idx, catalogue, number, starAlias):
    '''Looks for a catalogue name and ID in starAlias
    
    Parameters
    ----
    idx : int
        index of original catalogue array where catalogue and number come from

    catalogue : str(1)
        index of catalogue name ('S', 'M', etc) of the star to search

    number : int
        catalogue ID for the star to search

    starAlias : np.array [obj name, alias]
        Array with the full list of aliases for each star

    '''
    
    result = (np.nan, np.nan)
    for myIdx in range(starAlias.shape[0]):
        
        #retrieves the parsed data from the star alias list
        myStar, myCatalogue, myNumber = read_myStar(starAlias, myIdx)
#         print myCatalogue, catalogue
#         print myNumber, number
#         print

        if (myCatalogue==catalogue): #is it the same catalogue
#             print myCatalogue, catalogue
#             print myNumber, number
#             print type(myNumber), type(number)
    
            if (myNumber==number): #is it the same ID
#                 print myStar,'yes!!!!!!!',
                result = idx, myStar
                break

    return result
    

# <codecell>

#Creates an array of stars in the reference catalgue that are also in my catalogue [myName, other catalogue idx]

goodStars = []
total = 0
for i,cat in enumerate(geller.columns[1][:]):
    
#     print '>>>>',i,cat
    try:
        result = find_in_my_stars(i, cat[0], int(cat[1:]), starAlias)
#         print result
        if not np.isnan(result[0]):
            total += 1
#             print total,
            goodStars.append(result)
    except:
        pass
    
goodStars = np.array(goodStars)

# <codecell>

#subset of found stars in the original table
subGeller = geller[goodStars[:,0].astype(int)]

# <codecell>

#add myID column to the original table
col_c = table.Column(name='myName', data=goodStars[:,1])
subGeller.add_column(col_c, index =0)

# <codecell>

#full table print 
print subGeller.to_pandas().to_latex(index=False)

# <codecell>

#reduced column number
#'myName','IDW','IDX','RAh','RAm','RAs','DEd','DEm','DEs','Vmag','B-V','NW','NC','JD0','JDf','RVel','e_RVel','i','PRV','PPMy','PPMz','PPMg','PPMs','e/i','Pchi2','Class','Com' 
# print subGeller['myName','IDW','IDX','Vmag','B-V','RVel','Class','Com'].to_pandas().to_latex(index=False)
print subGeller['myName','IDW','IDX','Vmag','B-V','RVel','e_RVel','i','e/i','Class','Com'].to_pandas().to_latex(index=False)

