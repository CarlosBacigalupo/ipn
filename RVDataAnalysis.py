# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pandas as pd

# <codecell>

cd ~/Documents/HERMES/reductions/47Tuc_core_6.2/npy

# <codecell>

cd ~/Documents/HERMES/reductions/47Tuc_core_6.2/npy_150509/

# <codecell>

cd ~/Documents/HERMES/reductions/47Tuc_core_1arc_6.2/npy/

# <codecell>

cd ~/Documents/HERMES/reductions/47Tuc_core_1arc_6.2/npy_150509/

# <codecell>

cd ~/Documents/HERMES/reductions/HD1581_6.2/npy/

# <codecell>

cd ~/Documents/HERMES/reductions/HD285507_6.2/npy_150509/

# <codecell>

cd ~/Documents/HERMES/reductions/HD285507_6.2/npy/

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

data=np.load('data.npy')
RVs=np.load('RVs.npy')
sigmas=np.load('sigmas.npy')
baryVels=np.load('baryVels.npy')
JDs=np.load('JDs.npy')
thisSlice = RVs
totalStars = RVs.shape[0]
goodStars = np.sum(((thisSlice<5000) & (thisSlice!=0)), axis=0).astype(float)
size = np.array(thisSlice.shape)
total = np.reshape(np.repeat(size[0], size[1]*size[2]),(size[1],size[2]))
labels = ['Blue','Green','Red','IR']
a = pd.DataFrame(goodStars/total*100)
a.columns = labels

# <codecell>

print a.to_latex(formatters=[f1, f1, f1, f1])

# <codecell>

def f1(x):
    return '%5.2f' % x

# <codecell>

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

X = np.linspace(0,10,1000)
Y = np.cos(X*20)

ax1.plot(X,Y)
ax1.set_xlabel(r"Original x-axis: $X$")

new_tick_locations = np.array([.2, .5, 5])

def tick_function(X):
    V = 1/(1+X)
    return ["%.3f" % z for z in V]

ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(tick_function(new_tick_locations))
ax2.set_xlabel(r"Modified x-axis: $1/(1+X)$")
plt.show()

# <codecell>

MEANRA  =     5.02704168586666 / 00 20 06.49
MEANDEC =     -64.868515716065 / -64 52 06.7     
00 20 04.25995, -64 52 29.2549
0.1853248777409312, -1.1426331243230099,
thisStar.RA = np.degrees(0.087738428718276612)
thisStar.Dec = np.degrees(-1.1321689058299416)       

# <codecell>

import toolbox

# <codecell>

toolbox.sex2dec(00 ,20,04.25995)*15

# <codecell>

-toolbox.sex2dec(64 ,52,29.2549)

# <codecell>

import RVTools
reload(RVTools)
import pickle
import pylab as plt

# <codecell>

filename = '/Users/Carlos/Documents/HERMES/reductions/HD1581_6.2/obj/HD1581.obj'
filehandler = open(filename, 'r')
thisStar = pickle.load(filehandler)

# <codecell>

thisCam = thisStar.exposures.cameras[1]

# <codecell>

for i in range(200,300):
    lambda1,flux1, lambda2,flux2, CCCurve, p, x_mask, RV = RVTools.single_RVs_CC_t0(thisStar, cam = 2, t = 14, 
                                                                                    xDef=100, corrHWidth=i)

# <codecell>

i=0
plt.plot(thisCam.red_fluxes[i])
print thisCam.fileNames[i]
plt.show()

# <codecell>


# <codecell>


