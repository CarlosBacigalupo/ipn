
# coding: utf-8

# In[1]:

import pylab as plt


# In[2]:

import numpy as np


# In[6]:

import glob


# In[52]:

ls


# In[160]:

cd /Users/Carlos/Documents/HERMES/reductions/new_start_6.5/HD1581/herpy_out/3


# In[92]:

cam - epoch (0-4)
0-0 169
0-1 169
0-2 169
0-3 169
0-3 169

1-0 169
1-1 169
1-2 169
1-3 169
1-4 169

2-0 215
2-1 168
2-2 168
2-3 267
2-4 169

3-0 169
3-1 169
3-2 169
3-3 188
3-4 187



# In[161]:

fList = glob.glob("*.npy")


# In[188]:

#print obj
for thisFile in fList:
    if "4_obj" in thisFile:
        npy = np.load(thisFile)
        plt.plot(npy[187], label = thisFile)
plt.legend(loc=0)
plt.show()    


# In[48]:

#print arcs
for thisFile in fList:
    if "arc" in thisFile:
        npy = np.load(thisFile)
        plt.plot(npy[169], label = thisFile)
plt.legend(loc=0)
plt.show()    


# In[179]:

# fList [0,2,6,10, 16]  #arcs
# fList [1,3,4,5,7,8,9,11,12,13,14,15,17,18,19]
npy = np.load(fList[11])
for i,flux in enumerate(npy[150:200]):
    plt.plot(flux, label = 100+i)
plt.legend(loc=0)
plt.show()    


# In[130]:

fList


# In[ ]:



