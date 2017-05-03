
# coding: utf-8

# In[ ]:

import pyfits as pf


# In[ ]:

cd /Users/Carlos/Documents/HERMES/reductions/myherpy/HD1581/4_25aug/1/


# In[ ]:

flatFileName = '25aug10039.fits'


# In[ ]:

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



# In[ ]:

flat = openFile(flatFileName)


# In[ ]:

import pylab as plt
import numpy as np


# In[ ]:

myColumn = flat[110:410,100]
myRange = range(110,410)


# In[ ]:

plt.plot(myRange, myColumn)


# In[ ]:

plt.show()


# In[ ]:

islansX = np.array()


# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')

fig = plt.figure()
ax = fig.add_subplot(111)
# ax.plot(myColumn, Swdown, '-', label = 'Swdown')
# ax.plot(myColumn, Rn, '-', label = 'Rn')
ax.plot(myRange, myColumn, '-')
# ax.plot(myColumn, '-', label = 'Rn')
ax2 = ax.twinx()
# ax2.plot(myColumn, '-r', label = 'temp')

# ax.legend(loc=0)
# ax.grid()
ax.set_xlabel("Pixel Number")
ax.set_ylabel(r"Intensity")
ax2.set_ylabel(r"Number of Islands")
# ax2.set_ylim(0, 35)
ax.set_ylim(0,max(myColumn)+500)
ax2.set_ylim(0,max(myColumn)+500)
ax.set_xlim(min(myRange),max(myRange))

myIslands = np.array([200,400,500, 1000, 1300, 1800, 2000, 2400, 2800, 3500, 4000, 4900])
for i in myIslands:
    ax2.plot((min(myRange),max(myRange)),(i,i), '-g')


ax2.set_yticks(myIslands)
ax2.set_yticklabels([0, 3, 13, 30 ,30, 30, 30, 30, 24, 11, 3, 1])
plt.title("Emergence Peak Finding Algorithm")
plt.show()


# In[ ]:



