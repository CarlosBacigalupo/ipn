# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from scipy.optimize import leastsq
import numpy as np
from scipy.optimize import minimize
import pylab as plt

# <codecell>

a.shape[0]

# <codecell>

a = np.zeros(100)
b = np.zeros(100)
a[51]=1
b[48]=1
c = np.convolve(a,b, 'same')
ans = np.where(c==np.max(c))
print ans, (a.shape[0])/2. - ans[0][0]
plt.plot(a)
plt.plot(b)
plt.plot(c)
plt.show()

# <codecell>

y0 = [6., 2., 0., 0., 2., 6.]

# <codecell>

ax = np.array([-3,-2,-1,0,1,2])

# <codecell>

p =np.array([10.,-10.,22])

# <codecell>

diff(p, y0, ax)

# <codecell>

pf = minimize(diff, p, (y0,ax), method='nelder-mead', tol=10,options={'maxiter':10000})

# <codecell>

print pf

# <codecell>

pf = leastsq(diff, p, (y0,ax), full_output = True, diag = [10,4,2])
print p

# <codecell>

pf[2]

# <codecell>

def quad(x,a,b,c):
    return a*x**2+b*x+c

# <codecell>

def diff(p, y0, ax):
    return np.sum(y0-quad(ax,p[0],p[1],p[2]))

# <codecell>


# <codecell>

import pyfits as pf
import os
import pylab as plt

# <codecell>

os.chdir('/Users/Carlos/Documents/HERMES/reductions/m67_all_4cams/cam1')

# <codecell>

a = pf.open('18dec10033red.fits')

# <codecell>

b  = a[0]
c  = a[1]
d  = a[2]

# <codecell>

plt.plot(a[4].data)
plt.show()

# <codecell>

a[4].header.items()

# <codecell>

c.data

# <codecell>

d.data

# <codecell>

cd ~/Downloads/

# <codecell>

import numpy as np
# dtypes = {'names': ('name','spect_type','vmag','ra_deg','dec_deg','bv_color'),
#           'formats': ('S1','S1','S1','S1','S1','S1',)}
a = np.loadtxt('BrowseTargets.txt', delimiter = '|', skiprows=13, dtype=str, usecols=(1,2,3,4,5,6)  )

# <codecell>

a[0]

# <codecell>

import pylab as plt
plt.plot(a[:,3].astype(float),a[:,4].astype(float))
plt.show()

# <codecell>

# a[:,2]= a[:,2].astype(float)
# a[:,3] = a[:,3]
# a[:,3]= a[:,3].astype(float)
# a[:,4]= a[:,4].astype(float)
# a[:,5]= a[:,5].astype(float)
a[:,3].astype(int)

# <codecell>

a

# <codecell>


