
# coding: utf-8

# In[2]:

import pylab as plt
import numpy as np
import pyfits as pf


# In[3]:

cd /Users/Carlos/Documents/HERMES/reductions/myherpy/HD1581/4_25aug/1/


# In[4]:

flatFileName = '25aug10043.fits'


# In[5]:

def openFile(fileName):
    thisFile = pf.open(fileName)

    print thisFile[0].header['OBJECT']
    
    gain0_2000  = thisFile[0].header['RO_GAIN']
    gain2000_4000  = thisFile[0].header['RO_GAIN1']

    thisData = thisFile[0].data

#     bias0_2000 = np.median(thisData[3:2052,4099:-3])
#     bias2000_4000 = np.median(thisData[2059:-3,4099:-3])

#     thisData = thisData[:,:4095]

#     thisData[:2055] -= bias0_2000
#     thisData[2055:] -= bias2000_4000
    
    return thisData



# In[6]:

arc = openFile(flatFileName)


# In[8]:

plt.imshow(arc[0:300, 0:300])
plt.show()


# In[ ]:



