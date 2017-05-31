
# coding: utf-8

# In[7]:

#!/usr/bin/python
import pylab as plt
import myHerpyTools as MHT 
import os
import numpy as np
import sys
import time
import importlib


# In[15]:

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
reduceCam = np.arange(4)
targetFolder =''


# In[16]:

# dataset = 'HD1581'
dataset = 'rhoTuc'
# dataset = 'HD285507'
if 1==1: #len(sys.argv)>1:
# if len(sys.argv)>1:
#     dataset = sys.argv[1]
    try:
        thisDataset = importlib.import_module('data_sets.'+dataset)
    except:
        print 'Could not load dataset:',dataset         
        sys.exit()
    
    
    try:
        os.makedirs(baseFolder+'/'+dataset+'/herpy_out')
    except:
        pass

    try:
        os.makedirs(baseFolder+'/'+dataset+'/herpy_out/0')
        os.makedirs(baseFolder+'/'+dataset+'/herpy_out/1')
        os.makedirs(baseFolder+'/'+dataset+'/herpy_out/2')
        os.makedirs(baseFolder+'/'+dataset+'/herpy_out/3')
    except:
        pass

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
    print dataFolders
    if reduceSet==-1: 
        reduceSet = dataFolders
    else:
        reduceSet = dataFolders[reduceSet]
        


# In[13]:

reload(MHT)
makeTramLine=True

for thisCam in reduceCam[3:]:
    
    targetFolder = baseFolder + '/' + dataset + '/' + 'herpy_out' + '/' + str(thisCam)    
    print 
    print 'Working on:',targetFolder
    
    for thisSetIdx, thisSet in enumerate([s + '/' + str(thisCam+1) for s in reduceSet]):

        if thisSetIdx>-1:   #condition to run only some epochs for this camera
            
            print 'Current set:',thisSet 
            fileList = [filename_prfx[thisSetIdx] + str(thisCam+1) + str(name).zfill(4)+ '.fits' for name in thisDataset.ix_array[thisSetIdx]]
            print 'File List', fileList

            flatFileName = baseFolder + '/' + dataset + '/' + thisSet + '/' + fileList[0] #'0_20aug/1/20aug10034.fits'

            if makeTramLine==True:
                #flat
                print time.strftime('%X %x %Z'), '  Starting open'
                flat = MHT.openFile(flatFileName)
                print time.strftime('%X %x %Z'), '  Ending open'
                print
                print time.strftime('%X %x %Z'), '  Starting make_flat_flat'
                flat_flat = MHT.make_flat_flat(flat)
                print time.strftime('%X %x %Z'), '  Ending make_flat_flat'
                print
                print time.strftime('%X %x %Z'), '  Starting make_flat_flat_bin'
                flat_flat_bin = MHT.make_flat_flat_bin(flat_flat)
                print time.strftime('%X %x %Z'), '  Ending make_flat_flat_bin'
                print
                print time.strftime('%X %x %Z'), '  Starting make_fibre_centroids'
                fibre_centroids = MHT.make_fibre_centroids(flat_flat_bin)
                print time.strftime('%X %x %Z'), '  Ending make_fibre_centroids'
                print
                print time.strftime('%X %x %Z'), '  Starting make_fibrePolys'
                fibrePolys = MHT.make_fibrePolys(fibre_centroids)
                print time.strftime('%X %x %Z'), '  Ending make_fibrePolys'
                print
                print time.strftime('%X %x %Z'), '  Starting make_tramlines'
                tramlines = MHT.make_tramlines(fibre_centroids, fibrePolys)
                print time.strftime('%X %x %Z'), '  Ending make_tramlines'
                print
                print "Tramlines shape:", tramlines.shape


                #arc
                arcFileName = baseFolder + '/' + dataset + '/' + thisSet + '/' + fileList[1] # '0_20aug/1/20aug10052.fits'
                arc =  MHT.openFile(arcFileName)

                #find vertical shift and apply it to the tram lines
                shift = MHT.find_vertical_shift(flat, arc) #result to be subtracted to the tramlines (1st array in the CC...)
                tramlines_shifted = tramlines - shift

                #save the tramline shifted
                pre = str(thisSetIdx) + '_' + 'tlm_s1'
                post = 'cam' + str(thisCam+1)
                MHT.write_NPY(flatFileName, pre, post, tramlines_shifted, targetFolder)

            else:
                #arc
                arcFileName = baseFolder + '/' + dataset + '/' + thisSet + '/' + fileList[1] # '0_20aug/1/20aug10052.fits'
                arc =  MHT.openFile(arcFileName)
                
                pre = str(thisSetIdx) + '_' + 'tlm_s1'
                post = 'cam' + str(thisCam+1)
                tramlines_shifted = MHT.read_NPY(flatFileName, pre, post, targetFolder)
                
        
            #save the arc
            extracted_arc = MHT.extract(tramlines_shifted, arc)
            print "Arc shape:", extracted_arc.shape
            pre = str(thisSetIdx) + '_' + 'arc_s1'
            post = 'cam' + str(thisCam+1)
            MHT.write_NPY(arcFileName, pre, post, extracted_arc, targetFolder)


            #obs (1 or more)
            for thisFileIdx, thisFile in enumerate(fileList[2:]):
                objFileName = baseFolder + '/' + dataset + '/' + thisSet + '/' + thisFile # '0_20aug/1/20aug10053.fits'
                print 'Opening',objFileName

                obj =  MHT.openFile(objFileName)
                extracted_obj = MHT.extract(tramlines_shifted, obj)
                print "Obj shape:", extracted_obj.shape
                pre = str(thisSetIdx) + '_' + 'obj' + str(thisFileIdx+1) + '_s1' 
                post = 'cam' + str(thisCam+1)
                MHT.write_NPY(objFileName, pre, post, extracted_obj, targetFolder)

    #             for i in tramlines_shifted:
    #                 plt.plot(i)
    #             plt.imshow(obj)
    #             title = 'cam', thisCam, 'file',objFileName
    #             plt.title(title)
    #             plt.yticks(tramlines_shifted[:,0], range(tramlines_shifted.shape[0]))
    #             plt.show()
    #             print flatFileName, arcFileName, objFileName


# In[ ]:

# fig, ax1 = plt.subplots()

for i in tramlines_shifted:
    plt.plot(i)

plt.yticks(tramlines_shifted[:,0], range(tramlines_shifted.shape[0]))
plt.show()


# In[ ]:


    #run forest, run
    if booLog==True: sys.stdout = open(str(startFrom)+str(reduceSet)+'_'+str(time.strftime('%X'))+'.log', 'w')
             
    print time.strftime('%X %x %Z'), '  Starting reduction'
    print time.strftime('%X %x %Z'), '  Ending reduction'
    
    # n=0
    # for dataset, fileN in zip(filename_prfx, ix_array):
    #     for i in fileN[2:]:
    #         print str(n)+'_'+str(dataset),'\t',str(i)
    #     n+=1

else:
    print 'No data_set specified.'
    print 'run_myHerpy_1.py data_set [reduceSet] [reduceCam]'
    
    
    


# In[ ]:



