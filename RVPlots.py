# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pylab as plt
import numpy as np
import glob
import pickle

# <codecell>

def all_spec_overlap(thisStar, thisCam = '', booShow = True, booSave = False):
    '''
    All spectra for thisCam, or all cams if thisCam is skipped.
    '''
    fig, ax = plt.subplots(2,2, sharey='all')
    
    # ax.set_yticks(thisStar.exposures.JDs)
    # ax.set_ylim(np.min(thisStar.exposures.JDs)-1,np.min(thisStar.exposures.JDs)+1)

    if thisCam in range(4):
        camList = [thisCam]
    else:
        camList = range(4)
        
    for cam in camList:
        thisCam = thisStar.exposures.cameras[cam]
        fileNames =  thisCam.fileNames
        nFluxes = thisCam.wavelengths.shape[0]
        ax[0,0].set_yticks(np.arange(0,nFluxes))
        ax[0,0].set_ylim(-1,nFluxes)
    
        for i in np.arange(nFluxes):
                if np.sum(thisCam.wavelengths[i])>0:
                    d, f = thisCam.wavelengths[i], thisCam.red_fluxes[i]/np.median(thisCam.red_fluxes[i])
                    if cam ==0:
                        ax[0,0].plot(d, f+i, 'b')
                    elif cam==1:
                        ax[0,1].plot(d, f+i, 'g')
                    elif cam==2:
                        ax[1,0].plot(d, f+i, 'r')
                    elif cam==3:
                        ax[1,1].plot(d, f+i, 'cyan')
                #         ax.plot(d, f+thisStar.exposures.JDs[i], 'k')
    
    plt.xlabel('Wavelength [Ang]')
    plt.title(thisStar.name+' - Camera '+str(cam+1))
    plt.gca().set_yticks(range(thisCam.wavelengths.shape[0]))
    plt.gca().set_yticklabels(thisCam.fileNames)
#     ax[0,0].set_yticklabels(fileNames)
    if booSave==True: 
        try:
            plt.savefig('plots/' + objName[:-4] + 'SOvl_cam' + str(cam+1), dpi = 1000 )
        except:
            pass
    if booShow==True: plt.show()

# <codecell>

def RVs_single_star(thisStar, sigmaClip = -1, RVClip = -1, booSave = False, booShow = True):
    
#     #Prepare arrays
#     if sigmaClip >-1:
#         stdRV = 
        
        
        
        
#     RVs = thisCam.RVs
    
    
    #Plots RVs, baryvels. Single star, 4 cameras
    plt.title(thisStar.name)
    
    #Barycentric velocity
#     plt.plot(thisStar.exposures.JDs, thisStar.exposures.rel_baryVels, label = 'Barycentric Vel. ')


    #RVs from red_.obj files
    thisCam = thisStar.exposures.cameras[0]
    RVMask = thisCam.safe_flag
#     plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'Blue', color ='b' )
    plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.RVs[RVMask] + thisStar.exposures.rel_baryVels, yerr=thisCam.sigmas[RVMask],fmt='.', label = 'Blue', color ='b' )

    thisCam = thisStar.exposures.cameras[1]
    RVMask = thisCam.safe_flag
#     plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'Green' , color ='g')
    plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.RVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'Green' , color ='g')

    thisCam = thisStar.exposures.cameras[2]
    RVMask = thisCam.safe_flag
#     plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'Red' , color ='r')
    plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.RVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'Red' , color ='r')

    thisCam = thisStar.exposures.cameras[3]
    RVMask = thisCam.safe_flag
#     plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'IR', color ='cyan' )
    plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.RVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'IR', color ='cyan' )

    
    #Plot sine curve
    # start_day = 56889.000000 # The MJulian date for CE  2014 August 20 00:00:00.0 UT  (10am australia)
    # end_day = 56895.000000 #The MJulian date for CE  2014 August 26 00:00:00.0 UT (10am australia)

    # days = np.linspace(start_day, end_day) 

    # K1=26100
    # peri_time = 19298.85
    # P=4.8202
    # peri_arg=269.3
    # RV = K1* np.sin( (days-peri_time)/P*2*np.pi - peri_arg/360*2*np.pi )
    # plt.plot(days, RV, linewidth = 1, label = 'rhoTuc' )

    plt.xlabel('JD')
    plt.ylabel('RV [m/s]')
    plt.legend(loc=3)
    if booSave==True: 
        try:
            plt.savefig('plots/' + thisStar.name + '_RVs' + str(cam+1), dpi = 1000 )
        except:
            pass
    if booShow==True: plt.show()

# <codecell>

def RVs_all_stars(booSave = False, booShow = True):
    
    fileList = glob.glob('red_*.obj')

    if len(fileList)>0:
        print 'About to plot RVs from ',len(fileList),'stars.'
        for objName in fileList:
            print 'Ploting RVs from',objName
            filehandler = open(objName, 'r')
            thisStar = pickle.load(filehandler)

            #Plots RVs, baryvels. Single star, 4 cameras
            plt.title(thisStar.name)
            plt.plot(thisStar.exposures.JDs, -thisStar.exposures.rel_baryVels, label = 'Barycentric Vel. ')

            thisCam = thisStar.exposures.cameras[0]
            # RVMask = thisStar.exposures.my_data_mask
            # RVMask = ((thisStar.exposures.my_data_mask) & (np.abs(thisCam.RVs)<50000))
            RVMask = thisCam.safe_flag
        #     plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'Blue', color ='b' )
            plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.RVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'Blue', color ='b' )

            thisCam = thisStar.exposures.cameras[1]
            # RVMask = thisStar.exposures.my_data_mask
            # RVMask = np.abs(thisCam.RVs)<1e6
            RVMask = thisCam.safe_flag
        #     plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'Green' , color ='g')
            plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.RVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'Green' , color ='g')

            thisCam = thisStar.exposures.cameras[2]
            # RVMask = thisStar.exposures.my_data_mask
            # RVMask = np.abs(thisCam.RVs)<2e5
            RVMask = thisCam.safe_flag
        #     plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'Red' , color ='r')
            plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.RVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'Red' , color ='r')

            thisCam = thisStar.exposures.cameras[3]
            # RVMask = thisStar.exposures.my_data_mask
            # RVMask = ((thisStar.exposures.my_data_mask) & (np.abs(thisCam.RVs)<20000))
            RVMask = thisCam.safe_flag
        #     plt.scatter(thisStar.exposures.JDs[RVMask], thisCam.DDRVs[RVMask], label = 'IR', color ='cyan' )
            plt.errorbar(thisStar.exposures.JDs[RVMask], thisCam.RVs[RVMask], yerr=thisCam.sigmas[RVMask],fmt='.', label = 'IR', color ='cyan' )

            # start_day = 56889.000000 # The Julian date for CE  2014 August 20 00:00:00.0 UT  (10am australia)
            # end_day = 56895.000000 #The Julian date for CE  2014 August 26 00:00:00.0 UT (10am australia)

            # days = np.linspace(start_day, end_day) 

            # K1=26100
            # peri_time = 19298.85
            # P=4.8202
            # peri_arg=269.3
            # RV = K1* np.sin( (days-peri_time)/P*2*np.pi - peri_arg/360*2*np.pi )
            # plt.plot(days, RV, linewidth = 1, label = 'rhoTuc' )

            plt.xlabel('JD')
            plt.ylabel('RV [m/s]')
            plt.legend(loc=3)
            if booSave==True: 
                try:
                    plotName = 'plots/All_RVs'
                    print 'Attempting to save', plotName, 
                    plt.savefig(plotName, dpi = 1000 )
                    
                except:
                    print 'FAILED'
            if booShow==True: plt.show()
                
        

# <codecell>

def RVs_all_stars_NPYs( sigmaClip = -1, RVClip = -1, booSave = False, booShow = True):
    
    data=np.load('data.npy')
    RVs=np.load('RVs.npy')
    sigmas=np.load('sigmas.npy')
    JDs=np.load('JDs.npy')

    X = JDs
    Y = RVs
    print X,Y
    if RVClip>-1:Y[np.abs(Y)>RVClip] = np.nan
#     if sigmaClip>-1:Y[np.median(Y)>RVClip] = np.nan
    YERR = sigmas
    
    colors = ['b','g','r','cyan']
    labels = ['Blue','Green','Red','IR']
    
#     if len(fileList)>0:
    print 'About to plot RVs from ',RVs.shape[0],'stars.'
#         for objName in fileList:
#             print 'Ploting RVs from',objName
#             filehandler = open(objName, 'r')
#             thisStar = pickle.load(filehandler)

    #Plots RVs, baryvels. Single star, 4 cameras
    plt.title('RV - All Stars')
#     plt.plot(JDs, -thisStar.exposures.rel_baryVels, label = 'Barycentric Vel. ')
    
    for i in range(Y.shape[0])[:]:
        for cam in range(4)[:]:
            print i,cam
#             plt.errorbar(X, Y[i,:,cam], yerr=YERR[i,:,cam], fmt='.', label = labels[cam], color = colors[cam])
            plt.scatter(X, Y[i,:,cam], label = labels[cam], color = colors[cam])
    
    #sine plot
#     start_day = 56889.000000 # The Julian date for CE  2014 August 20 00:00:00.0 UT  (10am australia)
#     end_day = 56895.000000 #The Julian date for CE  2014 August 26 00:00:00.0 UT (10am australia)
#     days = np.linspace(start_day, end_day) 
#     K1=26100
#     peri_time = 19298.85
#     P=4.8202
#     peri_arg=269.3
#     RV = K1* np.sin( (days-peri_time)/P*2*np.pi - peri_arg/360*2*np.pi )
#     plt.plot(days, RV, linewidth = 1, label = 'rhoTuc' )

    plt.xlabel('MJD')
    plt.ylabel('RV [m/s]')
#     plt.legend(loc=0)
    if booSave==True: 
        try:
            plotName = 'plots/All_RVs'
            print 'Attempting to save', plotName, 
            plt.savefig(plotName, dpi = 1000 )

        except:
            print 'FAILED'
    if booShow==True: plt.show()
                
        

