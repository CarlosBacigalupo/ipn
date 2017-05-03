
# coding: utf-8

# In[58]:

# import os
import create_obj as cr_obj
reload(cr_obj)
import pickle
# import glob
# import pyfits as pf
# import numpy as np
# import sys
# import toolbox
# import importlib


# In[59]:

cd ~/Documents/HERMES/reductions/myherpy/HD1581/


# In[60]:

starNames = ['Giant01']
# starNames = ['ThXe']


# In[61]:

thisStar = cr_obj.star(starNames[0])


# In[62]:

thisStar.name = 'ThXe'
thisStar.type = 'star'
thisStar.type = 'arc'
thisStar.exposures = cr_obj.exposures()


# In[63]:

thisStar.exposures.load_exposures_myherpy(thisStar.name, 0)


# In[64]:

thisStar.exposures.calculate_baryVels(thisStar)


# In[65]:

# file_pi = open('obj/HD1581.obj', 'w') 
file_pi = open('obj/ThXe.obj', 'w') 
pickle.dump(thisStar, file_pi) 
file_pi.close()


# In[35]:

thisStar.exposures.rel_baryVels


# In[ ]:



