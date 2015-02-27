# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
This file holds the low level routines to run 2dfdr reductions in bulk.
There are 3 main defs, bias_reduce, auto_reduce, auto_reduce_one_arc.
'''

# <codecell>

import numpy as np
import os
import shutil
import subprocess 

# <codecell>

class dr2df():
    
    #flags
    copyFiles = True
    overwrite = False
    reduceMode = ''
    
    
    #general
    target_root = ''
    display = 'localhost:21.0'
    idxFile = ''
    final_dir = ''
    doReduce = False
    
    #arrays
    filename_prfx = '' #file prefixes (09jan, 12aug, ...)
    ix_array = '' #file numbers array 
    date_list = ''  #list of observing dates (140205,...)
    source_dir_array  = '' #fully formed source folders array 
    
    
    # one per set
    target_dir = ''
    file_ix = ''
    source_dir = ''
    
    
    
    def runReduction(self):
        #Runs the reduction
        
        ############### Copy bit
        if self.copyFiles==True:
            print 'Copying all data files...'
            for thisSetIx in range(len(self.filename_prfx)):
                
                print '   ' + self.filename_prfx[thisSetIx]
                
                self.target_dir = self.target_root + str(thisSetIx) + '_'+ self.filename_prfx[thisSetIx] +'/'
                self.file_ix = self.ix_array[thisSetIx]
                self.source_dir = self.source_dir_array[thisSetIx]
                
                if self.create_folders():
                    self.create_file_list(thisSetIx)
                    
                    #copy fileset x 4cams
                    for cam, camList in enumerate([self.files1, self.files2, self.files3, self.files4]):
                        for i in camList:
                            src = self.source_dir  + 'ccd_' + str(cam+1) + '/' + i
                            dst = self.target_dir + '' + str(cam+1) + '/' + i
        
                            if not os.path.exists(dst):
                                shutil.copy(src, dst)
                                print '      Copied '+ dst
        
            print 'End of file copy'
            print ''

            
        ############### Reduction bit
        if self.reduceMode=='one_arc':
            print 'Starting one-arc reduction'

            #first do the 1st reduction as usual
            
            self.target_dir = self.target_root + str(0) + '_'+ self.filename_prfx[0] +'/'
            self.file_ix = self.ix_array[0]
            self.create_file_list(0)

            if self.doReduce==True:
                print '  Reducing first frame', self.filename_prfx[0]
                self.reduce_all() 
                print '  First frame reduced', self.filename_prfx[0]
                print ''
            else:
                print 'Reduction flag turned off. All done.'
            
            #replace flats and arcs for datasets>0
            print '   Copying reduced flats and arcs to subsequent data sets'
            self.copy_flat_arc()
            
            #reduce datasets>0
            for i in range(1,len(self.ix_array)):
                self.target_dir = self.target_root + str(i) + '_'+ self.filename_prfx[i] +'/'
                self.file_ix = self.ix_array[i]
                self.create_file_list(i)
    
                for cam,j in enumerate([self.files1, self.files2, self.files3, self.files4]):
                    os.chdir(self.target_dir + str(cam+1) + '/')
                    self.reduce_science(self, cam, j)
                
    
    
    def create_file_list(self, thisSetIx = 0 ):
        # Creates the list of files to be reduced based on date, name and indices.        
        self.files1 =  [self.filename_prfx[thisSetIx] +'1' + str(name).zfill(4)+ '.fits' for name in self.file_ix]
        self.files2 =  [self.filename_prfx[thisSetIx] +'2' + str(name).zfill(4)+ '.fits' for name in self.file_ix]
        self.files3 =  [self.filename_prfx[thisSetIx] +'3' + str(name).zfill(4)+ '.fits' for name in self.file_ix]
        self.files4 =  [self.filename_prfx[thisSetIx] +'4' + str(name).zfill(4)+ '.fits' for name in self.file_ix]

        
        
    def create_folders(self):

        try:
            os.rmdir(self.target_dir)
        except OSError as ex:
            if ((ex.errno == 66) or (ex.errno == 17)):
                if self.overwrite==True:
                    print '>>>> Overwriting', self.target_dir
                    os.system('rm -r '+self.target_dir )
                else:
                    print 'Target folder', self.target_dir,' not empty. Overwrite is off. '
                    return False
        os.mkdir(self.target_dir)
        os.mkdir(self.target_dir+'1/')
        os.mkdir(self.target_dir+'2/')
        os.mkdir(self.target_dir+'3/')
        os.mkdir(self.target_dir+'4/')
                
        try:
            os.mkdir(self.final_dir+'cam1/')
            os.mkdir(self.final_dir+'cam2/')
            os.mkdir(self.final_dir+'cam3/')
            os.mkdir(self.final_dir+'cam4/')
        except:
            print '>>>> Final folder creation failed(no biggie), please check '+self.final_dir
        return True
    
    
            
    def reduce_all(self):
        
        # environment vars for os call if using subprocess
        env = {'PATH': self.dr_dir,'DISPLAY': self.display}
                
        #start reduction   
        for cam,j in enumerate([self.files1, self.files2, self.files3, self.files4]):
            os.chdir(self.target_dir + str(cam+1) + '/')
            print '      Flat ',j[0]
            print '      Arc ',j[1]
            print '      Science files ' + str(j[2:])

            
            #flat
            try:
                os.rmdir(self.target_dir + str(cam+1) + '/' + j[0][:-5]+'_outdir')
            except OSError as ex:
                if ex.errno == 66:
                    print "Target folder not empty."
                    return False

            print '      >>Reducing flat'    
            os.mkdir (j[0][:-5]+'_outdir')                   
            os_command =  'drcontrol'
            os_command += ' reduce_fflat ' + j[0]
#            if useBias==True: os_command += ' -BIAS_FILENAME BIAScombined.fits'
            os_command += ' -idxfile ' + self.idxFile
            os_command += ' -OUT_DIRNAME '  + j[0][:-5]+'_outdir'
            os.system('killall drcontrol')
            os.system('killall drexec')
            print '      OS Command '+ os_command
#             out = subprocess.call(os_command, env = env, shell = True)
            os.system(os_command)
    
    
            #arc                 
            try:
                os.rmdir(self.target_dir + str(cam+1) + '/' + j[1][:-5]+'_outdir')
            except OSError as ex:
                if ex.errno == 66:
                    print "Target folder not empty."
                    return False
                
            print '      >>Reducing arc'    
            os.mkdir(self.target_dir + str(cam+1) + '/' + j[1][:-5]+'_outdir')                   
            os_command =  'drcontrol'
            os_command += ' reduce_arc '  + j[1]
#             if useBias==True: os_command += ' -BIAS_FILENAME BIAScombined.fits'
            os_command += ' -idxfile ' + self.idxFile
            os_command += ' -TLMAP_FILENAME ' + j[0][:-5] + 'tlm.fits'
            os_command += ' -OUT_DIRNAME ' + j[1][:-5]+'_outdir'
            os.system('killall drcontrol')
            os.system('killall drexec')
            print '      OS Command '+ os_command
#             out = subprocess.call(os_command, env = env, shell = True)
            os.system(os_command)
#                         shutil.copyfile(j[self.flat][:-5] + 'tlm.fits', '../../cam' +str(cam)+'/'+ j[self.flat][:-5] + 'tlm.fits')


            #flat
            print '      >>Scrunching flat (whatever that means)'    
            os_command =  'drcontrol'
            os_command += ' reduce_fflat ' + j[0]
            os_command += ' -idxfile ' + self.idxFile
#             if useBias==True: os_command += ' -BIAS_FILENAME BIAScombined.fits'
            os_command += ' -WAVEL_FILENAME ' + j[1][:-5] + 'red.fits'
            os_command += ' -OUT_DIRNAME ' + j[0][:-5]+'_outdir'
            os.system('killall drcontrol')
            os.system('killall drexec')
            print '      OS Command '+ os_command
#             out = subprocess.call(os_command, env = env, shell = True)
            os.system(os_command)
            
            #THIS SHOULD CALL reduce_science
            #science 
            self.reduce_science(cam, j)
#             obj_files = np.array(j[2:])
#             for obj in obj_files:
#                 try:
#                     os.rmdir(obj[:-5]+'_outdir')
#                 except OSError as ex:
#                     if ex.errno == 66:
#                         print 'Target folder (', obj[:-5]+'_outdir', 'not empty.'
#                         return False

#                 print '      >>Reducing science '+ obj                        
#                 os.mkdir(obj[:-5]+'_outdir')                   
#                 os_command =  'drcontrol'
#                 os_command += ' reduce_object ' + obj
#                 os_command += ' -idxfile ' + self.idxFile
#                 os_command += ' -WAVEL_FILENAME ' + j[1][:-5] + 'red.fits'
# #                 if useBias==True: os_command += ' -BIAS_FILENAME BIAScombined.fits'
#                 os_command += ' -TLMAP_FILENAME ' + j[0][:-5] + 'tlm.fits'
#                 os_command += ' -FFLAT_FILENAME ' + j[0][:-5] + 'red.fits'
#                 os_command += ' -OUT_DIRNAME ' + obj[:-5]+'_outdir'
# #                                 os_command += ' -TPMETH OFFSKY'
#                 os.system('killall drcontrol')
#                 os.system('killall drexec')
#                 print '      OS Command '+ os_command
# #                                     out = subprocess.call(os_command, env = env, shell = True)
#                 os.system(os_command)
    
#                 print '      Copying '+ obj[:-5]+'red.fits to ' + self.final_dir + 'cam' + str(cam+1) + '/'               
#                 shutil.copyfile(obj[:-5]+'red.fits', self.final_dir + 'cam' + str(cam+1) + '/' + obj[:-5]+'red.fits')
# #                                     shutil.copyfile(obj, '../../cam' +str(cam)+'/'+ obj)
    
    
    
    

    def copy_flat_arc(self):     
        
        #crates the list of source files fro dataset=0
        self.target_dir = self.target_root + str(0) + '_'+ self.filename_prfx[0] +'/'
        self.file_ix = self.ix_array[0]
        self.create_file_list(0)
        src_list = []
        for cam,j in enumerate([self.files1, self.files2, self.files3, self.files4]):
            src_list.append(self.target_dir + str(cam+1) + '/' +j[0])
            src_list.append(self.target_dir + str(cam+1) + '/' +j[0][:-5] + 'ex.fits')
            src_list.append(self.target_dir + str(cam+1) + '/' +j[0][:-5] + 'im.fits')
            src_list.append(self.target_dir + str(cam+1) + '/' +j[0][:-5] + 'red.fits')
            src_list.append(self.target_dir + str(cam+1) + '/' +j[0][:-5] + 'tlm.fits')
            src_list.append(self.target_dir + str(cam+1) + '/' +j[1])
            src_list.append(self.target_dir + str(cam+1) + '/' +j[1][:-5] + 'ex.fits')
            src_list.append(self.target_dir + str(cam+1) + '/' +j[1][:-5] + 'im.fits')
            src_list.append(self.target_dir + str(cam+1) + '/' +j[1][:-5] + 'red.fits')
        
        #for all datasets>0 create target_list and copy src_list to target_list
        for i in range(1,len(self.ix_array)):
            self.target_dir = self.target_root + str(i) + '_'+ self.filename_prfx[i] +'/'
            self.file_ix = self.ix_array[i]
            self.create_file_list(i)
            target_list = []
            for cam,j in enumerate([self.files1, self.files2, self.files3, self.files4]):
                target_list.append(self.target_dir + str(cam+1) + '/' +j[0])
                target_list.append(self.target_dir + str(cam+1) + '/' +j[0][:-5] + 'ex.fits')
                target_list.append(self.target_dir + str(cam+1) + '/' +j[0][:-5] + 'im.fits')
                target_list.append(self.target_dir + str(cam+1) + '/' +j[0][:-5] + 'red.fits')
                target_list.append(self.target_dir + str(cam+1) + '/' +j[0][:-5] + 'tlm.fits')
                target_list.append(self.target_dir + str(cam+1) + '/' +j[1])
                target_list.append(self.target_dir + str(cam+1) + '/' +j[1][:-5] + 'ex.fits')
                target_list.append(self.target_dir + str(cam+1) + '/' +j[1][:-5] + 'im.fits')
                target_list.append(self.target_dir + str(cam+1) + '/' +j[1][:-5] + 'red.fits')
            
            #copy this set
            for src,tgt in zip(src_list, target_list):
                print 'Copying ',src,' to ',tgt
                shutil.copyfile(src,tgt)
            
            
    
    
    
    def reduce_science(self, cam, j):
        
        #science
        obj_files = np.array(j[2:])
        for obj in obj_files:
            try:
                os.rmdir(obj[:-5]+'_outdir')
            except OSError as ex:
                if ex.errno == 66:
                    print 'Target folder (', obj[:-5]+'_outdir', 'not empty.'
                    return False

            print '      >>Reducing science '+ obj                        
            os.mkdir(obj[:-5]+'_outdir')                   
            os_command =  'drcontrol'
            os_command += ' reduce_object ' + obj
            os_command += ' -idxfile ' + self.idxFile
            os_command += ' -WAVEL_FILENAME ' + j[1][:-5] + 'red.fits'
#                 if useBias==True: os_command += ' -BIAS_FILENAME BIAScombined.fits'
            os_command += ' -TLMAP_FILENAME ' + j[0][:-5] + 'tlm.fits'
            os_command += ' -FFLAT_FILENAME ' + j[0][:-5] + 'red.fits'
            os_command += ' -OUT_DIRNAME ' + obj[:-5]+'_outdir'
#                                 os_command += ' -TPMETH OFFSKY'
            os.system('killall drcontrol')
            os.system('killall drexec')
            print '      OS Command '+ os_command
#                                     out = subprocess.call(os_command, env = env, shell = True)
            os.system(os_command)
    
            print '      Copying '+ obj[:-5]+'red.fits to ' + self.final_dir + 'cam' + str(cam+1) + '/'               
            shutil.copyfile(obj[:-5]+'red.fits', self.final_dir + 'cam' + str(cam+1) + '/' + obj[:-5]+'red.fits')
#                                     shutil.copyfile(obj, '../../cam' +str(cam)+'/'+ obj)
    
    
    
    def bias_reduce(self, overwrite = False, copyFiles = True, idxFile = 'hermes.idx'):
        
        if (((copyFiles==True) and (self.create_folders(overwrite = overwrite))) or (copyFiles==False)):
                self.create_file_list()
            
                #copy files
                if copyFiles==True:
                    cam = 0
                    for camList in [self.files1, self.files2, self.files3, self.files4]:
                        cam += 1
                        for i in camList:
                            src = self.source_dir  + 'ccd_' + str(cam) + '/' + i
                            dst = self.target_dir + '' + str(cam) + '/' + i
                            if not os.path.exists(dst):
                                shutil.copy(src, dst)
                
                # environment vars for os call
                env = {'PATH': self.dr_dir,
                       'DISPLAY': self.display,
                       }
                
                            
                #start reduction
                cam = 0  
                out = 0      
                for j in [self.files1, self.files2, self.files3, self.files4]:
                    cam += 1
                    os.chdir(self.target_dir + str(cam) + '/')
                    print j
                    
                    for obj in j:
                        os_command =  'drcontrol'
                        os_command += ' reduce_bias ' + obj
                        os_command += ' -idxfile ' + idxFile
                        os.system('killall drcontrol')
                        os.system('killall drexec')
                        os.system(os_command)
   

# <codecell>


# <codecell>


