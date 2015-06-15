# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pylab as plt

# <markdowncell>

# radial velocity semi aplitude

# <codecell>

K0 = 28.4329

# <codecell>

pMass = 1
P = 6./365

# <codecell>

K = K0 * pMass * sMass**(-2./3.) * P**(-1./3.)

# <codecell>

P= np.arange(2,10,0.1)
K = K0 * pMass * sMass**(-2./3.) * (P/365)**(-1./3.)

# <codecell>

P= np.arange(2,30,0.1)
sMass = 0.5
K = K0 * pMass * sMass**(-2./3.) * (P/365)**(-1./3.)
plt.plot(P,K, label= str(sMass)+' M$_{sun}$ star')
sMass = 1
K = K0 * pMass * sMass**(-2./3.) * (P/365)**(-1./3.)
plt.plot(P,K, label= str(sMass)+' M$_{sun}$ star')
sMass = 1.5
K = K0 * pMass * sMass**(-2./3.) * (P/365)**(-1./3.)
plt.plot(P,K, label= str(sMass)+' M$_{sun}$ star')

plt.ylabel('RV [m/s]')
plt.xlabel('P (days)')
plt.title('RV signal for a 1 M$_{jup}$ planet')
plt.legend(loc=0)
plt.show()

# <codecell>


# <codecell>

P1 = np.random.random(10)*10
# sMass = 0.5
RV1 = 120*np.sin(P1)
yerr1 = 200+np.random.random(10)*500
# yerr1 = np.zeros(10)

plt.errorbar(P1,RV1, label= 'Simulated Data', c='r', fmt = 'o', yerr=yerr1)
P= np.arange(0,10,0.1)
RV = 120*np.sin(P)
plt.plot(P,RV, label='Fitted Curve')
# sMass = 1
# K = K0 * pMass * sMass**(-2./3.) * (P/365)**(-1./3.)
# plt.plot(P,K, label= str(sMass)+' M$_{sun}$ star')
# sMass = 1.5
# K = K0 * pMass * sMass**(-2./3.) * (P/365)**(-1./3.)
# plt.plot(P,K, label= str(sMass)+' M$_{sun}$ star')

plt.ylabel('RV [m/s]')
plt.xlabel('P (days)')
plt.title('Simulated RV signal for a 1 M$_{jup}$ planet')
plt.legend(loc=0)
plt.show()

# <codecell>

# P1 = np.random.random(10)*10
# sMass = 0.5
# RV1 = 40*np.sin(P1*2*np.pi)
yerr1 = 200+np.random.random(10)*500
# yerr1 = np.zeros(10)

plt.errorbar(P1,RV1, label= 'Simulated Data', c='r', fmt = 'o', yerr=yerr1)
P= np.arange(0,10,0.1)
RV = 40*np.sin(P*2*np.pi)
plt.plot(P,RV, label='Fitted Curve')
sMass = 1
# K = K0 * pMass * sMass**(-2./3.) * (P/365)**(-1./3.)
# plt.plot(P,K, label= str(sMass)+' M$_{sun}$ star')
# sMass = 1.5
# K = K0 * pMass * sMass**(-2./3.) * (P/365)**(-1./3.)
# plt.plot(P,K, label= str(sMass)+' M$_{sun}$ star')

plt.ylabel('RV [m/s]')
plt.xlabel('P (days)')
plt.title('Simulated oscillations for a 1.6 M$_{sum}$ star')
plt.legend(loc=0)
plt.show()

