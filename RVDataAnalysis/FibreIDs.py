# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=3>

# 2dfdr Fibre IDs Table and Index values. General info and per target list

# <codecell>

import pandas as pd
import numpy as np

# <codecell>

labels = ['FibreID', '2dfID (=fldID, =PIVOT)', 'idxData']

rev_num = np.tile(np.arange(10,0,-1),40)+np.repeat(np.arange(0,40)*10,10)

data = np.zeros((400,3))
# data[:] = ''
data[:,0] = np.arange(1,401)
data[:,1] = (rev_num)
data[:,2] = np.arange(0,400)
# print data
# data[range(49,399,50),5] = 'Guiding fibre'
a = pd.DataFrame(data)
a.columns = labels

pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 500)

# <codecell>

cd ~/Documents/HERMES/reductions/6.5/HD1581/0_20aug/1/

# <codecell>

ls

# <codecell>

import pyfits as pf
thisPf = pf.open('20aug10053.fits')

# <codecell>

thisPf[1].data[3]['PID']

# <codecell>

# ('NAME', 'S80'),
# ('RA', '>f8'), 
# ('DEC', '>f8'), 
# ('X', '>i4'), 
# ('Y', '>i4'), 
# ('XERR', '>i2'), 
# ('YERR', '>i2'), 
# ('THETA', '>f8'), 
# ('TYPE', 'S1'), 
# ('PIVOT', '>i2'), 
# ('MAGNITUDE', '>f8'), 
# ('PID', '>i4'), 
# ('COMMENT', 'S80'), 
# ('RETRACTOR', 'S10'), 
# ('WLEN', '>f8'), 
# ('PMRA', '>f8'), 
# ('PMDEC', '>f8')]

# <codecell>

print a.to_latex(index=False)

# <codecell>

# a.ix[np.hstack((range(60),range(350,400)))]
# a.ix[:20]

# <codecell>

# file_object = open('a.txt', 'w')
# file_object.write(a.to_latex(index=False))
# file_object.close()

# <codecell>


