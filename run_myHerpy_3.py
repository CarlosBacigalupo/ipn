
# coding: utf-8

# In[44]:

#!/usr/bin/python

import myHerpyTools as MHT 
import os
import numpy as np
import sys
import time
import importlib
import glob
import pylab as plt


# In[45]:

#reduction flags
# ver = '6.5'
booLog = False
useBias = False
copyFiles = False
doReduce = True
overwrite = False
# idxFile = 'no_flat_no_bias.idx'
startFrom = 0 #number of data set to begin with. 0 for beginning. Good for starting half way through if it cancelled
baseFolder='/Users/Carlos/Documents/HERMES/reductions/new_start_6.5'

# -1 for all
reduceSet = -1
reduceCam = np.array([0,1,3]) #don't have wl for 2(red)
targetFolder =''


# In[46]:

dataset = 'HD1581'
dataset = 'HD285507'
dataset = 'rhoTuc'
if 1==1: #len(sys.argv)>1:
# if len(sys.argv)>1:
#     dataset = sys.argv[1]
    try:
        thisDataset = importlib.import_module('data_sets.'+dataset)
    except:
        print 'Could not load dataset:',dataset         
        sys.exit()
    
#     if len(sys.argv)>2:
#         reduceSet = int(sys.argv[2])
#         if len(sys.argv)>3:
#             reduceCam = np.array([int(sys.argv[3])])
            
    #compose file prefixes from date_list
    months = np.array(['', 'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
    d = np.array([s[4:] for s in thisDataset.date_list])
    m = months[np.array([s[2:4] for s in thisDataset.date_list]).astype(int)]
    filename_prfx = np.core.defchararray.add(d, m)
    dataFolders = np.array([str(i)+'_'+s for i,s in enumerate(filename_prfx)])
    
    if reduceSet==-1: 
        reduceSet = dataFolders
    else:
        reduceSet = dataFolders[reduceSet]
        


# In[37]:

#HD1581
# 0-0 169
# 0-1 169
# 0-2 169
# 0-3 169
# 0-3 169

# 1-0 169
# 1-1 169
# 1-2 169
# 1-3 169
# 1-4 169

# 2-0 215
# 2-1 168
# 2-2 168
# 2-3 267
# 2-4 169

# 3-0 169
# 3-1 169
# 3-2 169
# 3-3 188
# 3-4 187

goodFibres = np.ones((4,5)).astype(int)*169
goodFibres[2,0] = 215
goodFibres[2,1] = 168
goodFibres[2,2] = 168
goodFibres[2,3] = 267
goodFibres[2,4] = 169

goodFibres[3,0] = 169
goodFibres[3,1] = 169
goodFibres[3,2] = 169
goodFibres[3,3] = 188
goodFibres[3,4] = 187


# In[38]:

# #HD285507
# cam (0-3) - epoch (0-4)
# 0-0 219
# 0-1 219
# 0-2 219
# 0-3 219
# 0-4 219

# 1-0 262
# 1-1 232
# 1-2 245
# 1-3 255
# 1-4 253


# # 2-0 215
# # 2-1 168
# # 2-2 168
# # 2-3 267
# # 2-4 169



# 3-0 218
# 3-1 220
# 3-2 220
# 3-3 218
# 3-4 218

goodFibres = np.ones((4,5)).astype(int)*219
goodFibres[1,0] = 262
goodFibres[1,1] = 232
goodFibres[1,2] = 245
goodFibres[1,3] = 255
goodFibres[1,4] = 253

goodFibres[3,0] = 218
goodFibres[3,1] = 220
goodFibres[3,2] = 220
goodFibres[3,3] = 218
goodFibres[3,4] = 218


# In[43]:

#rhoTuc
# cam (0-3) - epoch (0-4)
# 0-0 169
# 0-1 169
# 0-2 169
# 0-3 169 
# 0-4 169
# 0-5 169
# 0-6 168
# 0-7 168

# 1-0 169
# 1-1 169
# 1-2 169
# 1-3 169
# 1-4 169
# 1-5 169
# 1-6 168
# 1-7 168


# # 2-0 215
# # 2-1 168
# # 2-2 168
# # 2-3 267
# # 2-4 169



# 3-0 218
# 3-1 220
# 3-2 220
# 3-3 218
# 3-4 218

goodFibres = np.ones((4,8)).astype(int)*169
goodFibres[0,6] = 168
goodFibres[0,7] = 168

goodFibres[1,6] = 168
goodFibres[1,7] = 168

goodFibres[3,6] = 168
goodFibres[3,7] = 168


# In[47]:

reload(MHT)

for thisCam in reduceCam:
    
    targetFolder = baseFolder + '/' + dataset + '/' + 'herpy_out' + '/' + str(thisCam)    
    print targetFolder
    
#     #get the initial poly (wls wrt px) to find extended linelist
#     initialWlsFileName = baseFolder + '/initial_wls_' + str(thisCam) + '.txt'
#     initialPoly = MHT.make_poly_from_wls(initialWlsFileName)
    
    #this needs to be the fibre# loop....TODO
    thisFibreIdx = goodFibres[thisCam,0]
    finalLineListFileName = targetFolder + '/final_linelist_' + str(thisCam) + '_fib' +str(thisFibreIdx) + '.txt'
    finalLineList = np.loadtxt(finalLineListFileName)


    pxValuesEpoch = MHT.px_change_across_epochs(finalLineList, glob.glob1(targetFolder,"*arc*"), goodFibres[thisCam], targetFolder)

    
    
    order = 3
    epochPolyFit = []
    epochDiffToFit = []
    #the offsets of 1 epoch per loop
    for i in range(pxValuesEpoch.shape[1]):

        polyFilter = [(pxValuesEpoch[:,i]-pxValuesEpoch[:,0])<100]
#         print pxValuesEpoch[:,i]
        thisEpochPolyFit = np.polyfit(pxValuesEpoch[:,0],(pxValuesEpoch[:,i]-pxValuesEpoch[:,0]), order)
#         thisEpochDiffToFit = np.polynomial.polynomial.polyval(pxValuesEpoch[:,0], thisEpochPolyFit[::-1])-(pxValuesEpoch[:,i]-pxValuesEpoch[:,0])#     plt.plot(pxValuesEpoch[:,0],pxValuesEpoch[:,i]-pxValuesEpoch[:,0], '.')
        
#         plt.plot(pxValuesEpoch[:,0],np.polynomial.polynomial.polyval(pxValuesEpoch[:,0], thisEpochPolyFit[::-1]))
#         plt.plot(pxValuesEpoch[:,0],thisEpochDiffToFit, '.')
#         plt.show()
        
#         print thisEpochPolyFit
        epochPolyFit.append(thisEpochPolyFit)
#         epochDiffToFit.append(thisEpochDiffToFit)

        
        
        
    initialWlsFileName = baseFolder + '/initial_wls_' + str(thisCam) + '.txt'
    refWls = np.loadtxt(initialWlsFileName)

#         print epochPolyFit
    _,wls = MHT.make_WLS_from_polys(refWls, epochPolyFit)
        
        
        
        
#     print "WLS",wls
    
    #load extracted obj and arc and write with the new wls
    for thisSetIdx, thisSet in enumerate([s + '/' + str(thisCam+1) for s in reduceSet]):

        thisFibreIdx = goodFibres[thisCam, thisSetIdx]
        print thisFibreIdx
#         print thisSet 
        fileList = [filename_prfx[thisSetIdx] + str(thisCam+1) + str(name).zfill(4)+ '.fits' for name in thisDataset.ix_array[thisSetIdx]]

        arcFileName=fileList[1] 
        pre = str(thisSetIdx) + '_' + 'arc_s1'
        post = 'cam' + str(thisCam+1)
        extracted_arc = MHT.read_NPY(arcFileName, pre, post, targetFolder)

        for fileIdx, thisFileName in enumerate(fileList[2:]):
            objFileName=thisFileName
            pre = str(thisSetIdx) + '_obj' + str(fileIdx+1) + '_s1'
            post = 'cam' + str(thisCam+1)
            extracted_obj = MHT.read_NPY(objFileName, pre, post, targetFolder)

            if thisCam==10:
                for j in range(thisFibreIdx-10, thisFibreIdx+10):
                    plt.plot(extracted_obj[j], label=str(j))
                    plt.legend(loc=0)
                title = 'Cam:', thisCam,'fib',thisFibreIdx, 'file', objFileName
                plt.title(title)
                plt.show()
            print "wls shape", wls[thisSetIdx].shape
            print "arc shape", extracted_arc.shape
            print "obj shape", extracted_obj.shape
    #         print np.vstack((wls[thisSetIdx], extracted_arc[thisFibreIdx]))
            MHT.write_flux_and_wls(extracted_arc[thisFibreIdx], extracted_obj[thisFibreIdx], wls, thisSetIdx, fileIdx+1, targetFolder)
        
        
        
        
        
#     # get the flux from a single arc
#     objectArc = extracted_arc[thisFibre].copy() 
    
#     # Template to build model from.
# #     lineListfFileName = '../linelist_blue_v3.txt'
        
#     #Create the model, etc
#     thisPoly, thisSolution, thisErr = MHT.make_poly_model_err(objectArc, finalLineListFileName)

#     for thisSetIdx, thisSet in enumerate([s + '/' + str(thisCam+1) for s in reduceSet]):
        
#         print thisSet 
#         fileList = [filename_prfx[thisSetIdx] + str(thisCam+1) + str(name).zfill(4)+ '.fits' for name in thisDataset.ix_array[thisSetIdx]]
#         arcFileName=fileList[1] 
        
#         if thisSetIdx==0:
#             #open arc from epoch 0
#             pre = str(thisSetIdx) + '_' + 'arc_s1'
#             post = 'cam' + str(thisCam+1)

#             extracted_arc = MHT.read_NPY(arcFileName, pre, post, targetFolder)

#             #extend line list to all peaks
#             #creates the new linelist. Wrong, but equally wrong
#             extendedLinelist = MHT.extend_lineList(extracted_arc[thisFibreIdx],initialPoly)
            
            
#             pxValuesEpoch = MHT.px_change_across_epochs(extendedLinelist, glob.glob1(targetFolder,"*arc*"), thisFibreIdx, targetFolder)
            
#             #Calculate differences to fit in order to filter
#             order = 3
#             epochPolyFit = []
#             epochDiffToFit = []
#             #the offsets of 1 epoch per loop
#             for i in range(pxValuesEpoch.shape[1]):
#                 polyFilter = [(pxValuesEpoch[:,i]-pxValuesEpoch[:,0])<10]
#                 thisEpochPolyFit = np.polyfit(pxValuesEpoch[:,0][polyFilter],(pxValuesEpoch[:,i]-pxValuesEpoch[:,0])[polyFilter], order)
#                 thisEpochDiffToFit = np.polynomial.polynomial.polyval(pxValuesEpoch[:,0], thisEpochPolyFit[::-1])-(pxValuesEpoch[:,i]-pxValuesEpoch[:,0])
#                 epochPolyFit.append(thisEpochPolyFit)
#                 epochDiffToFit.append(thisEpochDiffToFit)

#             fibreFilter = np.sum(np.abs(epochDiffToFit)>0.4, axis=0)
# #             print lineList_v2[-fibreFilter.astype(bool)].shape

#             finalLineListFileName = targetFolder + '/final_linelist_' + str(thisCam) + '_fib' +str(thisFibreIdx) + '.txt'
#             np.savetxt(finalLineListFileName, extendedLinelist[-fibreFilter.astype(bool)])



# In[ ]:



