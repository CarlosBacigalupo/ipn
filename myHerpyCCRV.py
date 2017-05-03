
# coding: utf-8

# In[ ]:

import numpy as np


# In[ ]:

cd /Users/Carlos/Documents/HERMES/reductions/myherpy/HD1581/cam1/


# In[ ]:

ls


# In[ ]:

#Load the thxe arrays.
#pack them into a cube [exp, px, [wl,flux]] [15,4095,2]
#load the days (mayeb for use in interpolating the rv)

Th1 = np.loadtxt("ThXe_0_53.56889811.txt")
Th2 = np.loadtxt("ThXe_1_41.56890804.txt")
Th3 = np.loadtxt("ThXe_1_42.56890807.txt")
Th4 = np.loadtxt("ThXe_1_43.56890809.txt")
Th5 = np.loadtxt("ThXe_2_36.56891707.txt")
Th6 = np.loadtxt("ThXe_2_37.56891709.txt")
Th7 = np.loadtxt("ThXe_2_38.56891711.txt")
Th8 = np.loadtxt("ThXe_3_58.56893765.txt")
Th9 = np.loadtxt("ThXe_3_59.56893767.txt")
Th10 = np.loadtxt("ThXe_3_60.56893768.txt")
Th11 = np.loadtxt("ThXe_3_61.56893769.txt")
Th12 = np.loadtxt("ThXe_3_62.56893771.txt")
Th13 = np.loadtxt("ThXe_4_44.56894743.txt")
Th14 = np.loadtxt("ThXe_4_45.56894745.txt")
Th15 = np.loadtxt("ThXe_4_46.56894746.txt")

# Th1 = np.loadtxt("HD1581_0_53.56889811.txt")
# Th2 = np.loadtxt("HD1581_1_41.56890804.txt")
# Th3 = np.loadtxt("HD1581_1_42.56890807.txt")
# Th4 = np.loadtxt("HD1581_1_43.56890809.txt")
# Th5 = np.loadtxt("HD1581_2_36.56891707.txt")
# Th6 = np.loadtxt("HD1581_2_37.56891709.txt")
# Th7 = np.loadtxt("HD1581_2_38.56891711.txt")
# Th8 = np.loadtxt("HD1581_3_58.56893765.txt")
# Th9 = np.loadtxt("HD1581_3_59.56893767.txt")
# Th10 = np.loadtxt("HD1581_3_60.56893768.txt")
# Th11 = np.loadtxt("HD1581_3_61.56893769.txt")
# Th12 = np.loadtxt("HD1581_3_62.56893771.txt")
# Th13 = np.loadtxt("HD1581_4_44.56894743.txt")
# Th14 = np.loadtxt("HD1581_4_45.56894745.txt")
# Th15 = np.loadtxt("HD1581_4_46.56894746.txt")

Days = np.array([56889811,56890804,56890807,56890809,56891707,56891709,
                 56891711,56893765,56893767,56893768,56893769,56893771,56894743,
                 56894745,56894746])
ThCube = np.array([Th1, Th2, Th3, Th4, Th5, Th6, Th7, Th8, Th9, Th10, Th11, Th12, Th13, Th14, Th15])


# #### The the maximum comon wl range

# In[ ]:

np.max(ThCube[:,0,0]), np.max(ThCube[:,0,0])+0.00001


# In[ ]:

#added the +0.000001 because it flops on the interpolation otherwise ( the 12th decimal place!!!)roundup...
wlRange = np.array([np.max(ThCube[:,0,0])+0.00001, np.min(ThCube[:,-1,0])]) 


# In[ ]:

#Log axis of the common range (logged) 
#step size (Duncan's suggestion) is about 1500m/s (i.e. x2 upscaled)
logAxis = np.arange(np.log(wlRange)[0], np.log(wlRange)[1], 5*10**(-6))


# In[ ]:

#This holds the linearized interpolated flux for the 15 epoch
#all share the same wl bins, that's why we don't need that dimension in the array (loosing the 3rd dimension)
linLogThCube = np.zeros((15, logAxis.shape[0]))


# ### We re-sample the flux based on the linearised log axis. 

# In[ ]:

from scipy.interpolate import interp1d


# In[ ]:

np.save("ThXe_HD1851", linLogThCube)


# In[ ]:

for i in range(ThCube.shape[0]):
    f2 = interp1d(ThCube[i,:,0], ThCube[i,:,1], kind='cubic')
    linLogThCube[i,:] = f2(np.exp(logAxis))
    print i
    #this was used to check the out of range in the inerpolation
#     print np.exp(logAxis)[0], np.exp(logAxis)[-1]
#     print ThCube[i,:,0][0], ThCube[i,:,0][-1]
#     print np.exp(logAxis)[0]-ThCube[i,:,0][0], np.exp(logAxis)[-1]-ThCube[i,:,0][-1]


# In[ ]:

legs = np.arange(-(linLogThCube.shape[1]-1), linLogThCube.shape[1])


# In[ ]:

from scipy import constants

#the value of a pixel in m/s
valStep = (np.exp(logAxis)[9]-np.exp(logAxis)[8])/np.exp(logAxis)[8]*constants.c


# In[ ]:

#this is an axis of the 2*size of each spectrum in linLogThCube in km/s
valAxis = valStep * legs /1000


# In[ ]:

#mask to slice the CCResult for gaussian fitting
W = ((valAxis > -25) &  (valAxis < 25))
W = ((valAxis > -10) &  (valAxis < 10))


# In[ ]:

from scipy import optimize

def gaussian(x, mu, sig, ):
    x = np.array(x)
    return np.exp(-np.power(x - mu, 2.) / 2 / np.power(sig, 2.))


def flexi_gaussian(x, mu, sig, power, a, d ):
    x = np.array(x)
    return a* np.exp(-np.power(np.abs((x - mu) * np.sqrt(2*np.log(2))/sig),power))+d

def fit_gaussian(p, flux, x_range):
    a = optimize.leastsq(diff_gaussian, p, args= [flux, x_range])
    return a

def fit_flexi_gaussian(p, flux, x_range):
    a = optimize.leastsq(diff_flexi_gaussian, p, args= [flux, x_range])
    return a

def diff_gaussian(p, args):
    
    flux = args[0]
    x_range = args[1]

    diff = gaussian(x_range, p[0],p[1]) - flux
    return diff

def diff_flexi_gaussian(p, args):
    
    flux = args[0]
    x_range = args[1]
    weights = np.abs(np.gradient(flux)) * (flux+np.max(flux)*.1)
    diff = (flexi_gaussian(x_range, p[0], p[1], p[2], p[3], p[4]) - flux)# *weights
    return diff


# In[ ]:

for i in range(linLogThCube.shape[0])[:]:
#     plt.plot(logAxis,linLogThCube[i,:]/np.percentile(linLogThCube[i,:], 90))
    plt.plot(logAxis,linLogThCube[i,:])
plt.show()


# In[ ]:

import pylab as plt

flux1 =linLogThCube[0,:] # the reference flux
RVs = np.zeros((linLogThCube.shape[0]))

for i in range(linLogThCube.shape[0]): #range (the amount of epochs)
    flux2 =linLogThCube[i,:]

    ccResult = np.correlate(flux1, flux2, "full")


    x = valAxis[W]
    y = ccResult[W]
    y /= np.max(y)

    p,_ = fit_flexi_gaussian([0., 10., 2. ,1., 0.], y, x )
    
    plt.plot(x,y)
    plt.plot(x,flexi_gaussian(x, p[0], p[1], p[2], p[3], p[4]))
    plt.show()
    print p[0]
    



# In[ ]:

#ccs from star with +-25

6.14367202446e-09
-0.167474516269
-0.185552242756
-0.155002606345
-0.104232244195
-0.082658264156
-0.0747539297579
-0.747097050087
-0.751346051391
-0.744659958864
-0.780037056979
-0.752506902652
-1.33462563626
-1.33961076013
-1.35122423503


#     -8.6101706948e-10
#     -0.0623372536655
#     -0.0623372438661
#     -0.0623372741398
#     -0.011374196462
#     -0.0113741614478
#     -0.0113741019891
#     -0.187732113388
#     -0.187732103719
#     -0.187732110375
#     -0.187732104556
#     -0.187732113388
#     -0.910444964683
#     -0.91044496166
#     -0.910444965535

# In[ ]:

#to see the fitted gaussian
plt.plot(x, y)
plt.plot(x,flexi_gaussian(x, p[0], p[1], p[2], p[3], p[4]) )
plt.show()

