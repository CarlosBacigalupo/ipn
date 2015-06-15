# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pylab as plt
import numpy as np
import glob
import pickle
import scipy.stats as stats

# <codecell>

def all_spec_overlap(thisStar, thisCamIdx = '', booShow = True, booSave = False):
    '''
    All spectra for thisCam, or all cams if thisCam is skipped.
    '''
    fig, ax = plt.subplots(2,2, sharey='all')
    
    # ax.set_yticks(thisStar.exposures.JDs)
    # ax.set_ylim(np.min(thisStar.exposures.JDs)-1,np.min(thisStar.exposures.JDs)+1)

    if thisCamIdx in range(4):
        camList = [thisCamIdx]
    else:
        camList = range(4)
        
    for cam in camList:
        thisCam = thisStar.exposures.cameras[cam]
        fileNames =  thisCam.fileNames
        nFluxes = thisCam.wavelengths.shape[0]
#         ax[0,0].set_yticks(np.arange(0,nFluxes))
#         ax[0,0].set_ylim(-1,nFluxes)
    
        for i in np.arange(nFluxes):
                if np.sum(thisCam.wavelengths[i])>0:
                    d, f = thisCam.wavelengths[i], thisCam.red_fluxes[i]/np.median(thisCam.red_fluxes[i])
                    if cam ==0:
                        ax[0,0].plot(d, f+i*10, 'b')
                    elif cam==1:
                        ax[0,1].plot(d, f+i, 'g')
                    elif cam==2:
                        ax[1,0].plot(d, f+i, 'r')
                    elif cam==3:
                        ax[1,1].plot(d, f+i, 'cyan')
                #         ax.plot(d, f+thisStar.exposures.JDs[i], 'k')
    
    plt.xlabel('Wavelength [Ang]')
    plt.title(thisStar.name+' - Camera '+str(cam+1))
#     plt.gca().set_yticks(range(thisCam.wavelengths.shape[0]))
#     plt.gca().set_yticklabels(thisCam.fileNames)

#     ax.set_yticklabels(thisCam.fileNames)
#     ax.set_yticks(range(thisCam.wavelengths.shape[0]))

    
#     plt.xticks(rotation=70)

#     ax[0,0].set_yticklabels(fileNames)
    if booSave==True: 
        try:
            plt.savefig('plots/' + thisStar.name + 'SOvl_cam' + str(cam+1), dpi = 1000 )
        except:
            print 'Couldn''t save'
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

def RVs_all_stars_NPYs(idStars = [], sigmaClip = -1, RVClip = -1, topStars = -1, booSave = False, booShow = True, booBaryPlot = False, booBaryCorrect = False, title = ''):
    
    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
    sigmas=np.load('npy/sigmas.npy')
    baryVels=np.load('npy/baryVels.npy')
    JDs=np.load('npy/JDs.npy')
    
#     order = np.argsort(np.nanstd(RVs,axis=1),axis=0)
#     np.save('npy/order.npy',order)
    
    X = JDs
    Y = RVs
    
    if RVClip>-1:Y[np.abs(Y)>RVClip] = np.nan
    if sigmaClip>-1:
        stdY= np.std(Y)
        medY = np. median(Y)
        Y[(Y>=medY-sigmaClip*stdY) & (Y<=medY+sigmaClip*stdY)] = np.nan
    
    YERR = sigmas
    
    colors = ['b','g','r','cyan']
    labels = ['Blue','Green','Red','IR']
    
#     try:
#         order=np.load('order.npy')
#     except:
#         pass
    
    if topStars==-1:
        print 'Including all stars',Y.shape[0]
        topStars = Y.shape[0]
        
#     if len(fileList)>0:
    print 'About to plot RVs from '+str(topStars)+'/'+str(RVs.shape[0])+' stars.'
#         for objName in fileList:
#             print 'Ploting RVs from',objName
#             filehandler = open(objName, 'r')
#             thisStar = pickle.load(filehandler)

    try:
        Y[:,:,cam] = Y[:,:,cam][order[:,cam]]
        YERR[:,:,cam] = YERR[:,:,cam][order[:,cam]]
    except:
        pass
    
    for cam in range(4)[:]:
        for i in range(topStars)[:]:
            #Plots RVs, baryvels. Single star, 4 cameras
            
#             fig = plt.figure()
#             ax1 = fig.add_subplot(111)
            
            if title=='':
                plt.title('RV - '+str(topStars)+'/'+str(RVs.shape[0])+' - '+labels[cam]+' camera')
            else:
                plt.title(title, y=1.1)
            
            if data[i,0] in idStars:
                m = '+'
                s=100
                c='k'
            else:
                m = 'o'
                s=1
                c=colors[cam]
                
            
#             plt.errorbar(X, Y[i,:,cam], yerr=YERR[i,:,cam]*1000, fmt='.', label = labels[cam], color = colors[cam])
            if booBaryCorrect==True: 
                plt.scatter(X, Y[i,:,cam]- baryVels, label = labels[cam], color = c, s=s, marker=m)

            else:
                plt.scatter(X, Y[i,:,cam], label = labels[cam], color = c, s=s, marker=m)

            
#             plt.plot(X, Y[i,:,cam], label = labels[cam], color = colors[cam])
#             plt.scatter(X, Y[i,:,cam], label = labels[cam], color = 'k')

#             print 'RV', Y[i,:,cam]
#             print 'X', X
#             print 'bary', baryVels
#             print 'RV-bary median', np.median(Y[i,:,cam]-baryVels)
#             print 'RV-bary std', np.std(Y[i,:,cam]-baryVels)
        
        
        #sine plot
#         start_day = 56889.000000 # The Julian date for CE  2014 August 20 00:00:00.0 UT  (10am australia)
#         end_day = 56895.000000 #The Julian date for CE  2014 August 26 00:00:00.0 UT (10am australia)
#         days = np.linspace(start_day, end_day) 
#         K1=26100
#         peri_time = 19298.85
#         P=4.8202
#         peri_arg=269.3
#         RV = K1* np.sin( (days-peri_time)/P*2*np.pi - peri_arg/360*2*np.pi )
#         plt.plot(days, RV, linewidth = 1, label = 'rhoTuc' )

        if booBaryPlot==True: plt.plot(X, baryVels, 'k--', label = 'Barycentric Vel.')

#         plt.tight_layout()

        plt.xlabel('MJD')
        plt.ylabel('RV [m/s]')
        
        
#         ax2 = ax1.twiny()
#         ax2.set_xticks(X-X[0])
#         ax2.set_xticklabels(range(len(X)))
#         ax2.set_xlabel('Epoch')
        

    #     plt.legend(loc=0)
        if booSave==True: 
            try:
                plotName = 'plots/All_RVs_'+labels[cam]+''
                print 'Attempting to save', plotName
                plt.savefig(plotName)

            except:
                print 'FAILED'
        if booShow==True: plt.show()
        plt.close()
                
        

# <codecell>

def RVs_by_star_NPYs(sigmaClip = -1, RVClip = -1, booSave = False, booShow = True, booBaryPlot = False, booBaryCorrect = False, title = ''):
    
    data=np.load('data.npy')
    RVs=np.load('RVs.npy')
    sigmas=np.load('sigmas.npy')
    baryVels=np.load('baryVels.npy')
    JDs=np.load('JDs.npy')

    X = range(data[:,0].shape[0])
    Y = RVs
    Y[Y==0.]=np.nan
    
    if RVClip>-1:Y[np.abs(Y)>RVClip] = np.nan
    if sigmaClip>-1:
        stdY= np.std(Y)
        medY = np.median(Y)
        Y[(Y>=medY-sigmaClip*stdY) & (Y<=medY+sigmaClip*stdY)] = np.nan
    
    YERR = sigmas

    order = np.argsort(np.nanstd(RVs,axis=1),axis=0)
    np.save('order.npy',order)
    
    
    colors = ['b','g','r','cyan']
    labels = ['Blue','Green','Red','IR']
    
    #Plots RVs, baryvels. all star, 4 cameras
    print 'About to plot RVs from ',RVs.shape[0],'stars.'
    
    for cam in range(4)[:]:
        fig, ax = plt.subplots()

        ax.set_xticklabels(data[:,0])
        ax.set_xticks(X)
        plt.xticks(rotation=70)

        if booBaryPlot==True: plt.plot(X, baryVels, label = 'Barycentric Vel. ')

        if title=='':
            plt.title('RVs per star - '+labels[cam]+' camera')
        else:
            plt.title(title)

        Y[:,:,cam] = Y[:,:,cam][order[:,cam]]
        YERR[:,:,cam] = YERR[:,:,cam][order[:,cam]]
        
        #median
        plt.scatter(X, stats.nanmedian(Y[:,:,cam], axis = 1), label = labels[cam], color = 'k')
        
        #sigma
        plt.scatter(X, stats.nanmedian(Y[:,:,cam], axis = 1)+stats.nanstd(Y[:,:,cam], axis = 1), label = labels[cam], color = 'r')
        plt.scatter(X, stats.nanmedian(Y[:,:,cam], axis = 1)-stats.nanstd(Y[:,:,cam], axis = 1), label = labels[cam], color = 'r')
        
        
        #min max
        for star in range(Y.shape[0]):
            x = np.nanmax(Y[star,:,cam])
            n = np.nanmin(Y[star,:,cam])
            plt.plot([star,star],[x,n], color = 'g', lw=2)
            print star, n, x
        
        plt.xlabel('MJD')
        plt.ylabel('RV [m/s]')
    #     plt.legend(loc=0)

        if booSave==True: 
            try:
                plotName = 'plots/All_RVs_'+labels[cam]
                print 'Attempting to save', plotName
                plt.savefig(plotName)

            except:
                print 'FAILED'
        if booShow==True: plt.show()
        plt.close()        
        

# <codecell>

def SNR_W(RVCorrMethod = 'PM', thisStarName = 'Giant01', sigmaClip = -1, RVClip = -1, booSave = False, booShow = True, booBaryPlot = False, booBaryCorrect = False, title = ''):

    data=np.load('npy/data.npy')
    SNRs=np.load('npy/SNRs.npy')
    if RVCorrMethod=='PM':
        allW = np.load('npy/allW_PM.npy')
    else:
        allW = np.load('npy/allW_DM.npy')        

    idx = np.where(data[:,0]==thisStarName)[0][0]

    for cam in range(4):
        
        W = allW[:,cam,idx]

        thisSNRs = SNRs[:,0,cam]

        plt.plot(thisSNRs, label= 'SNR')
        plt.plot(W*np.nanmax(thisSNRs), label = 'W')
        plt.legend(loc=0)
        if title=='':
            title ='SNR and W - '+RVCorrMethod+' method - '+data[idx,0]+' - '+labels[cam]+' camera'            
        plt.title(title)
        plt.grid(True)
        plt.savefig(('PM_'+str(cam)))
        plt.show()

        if booSave==True: 
            try:
                plotName = 'plots/SNR_W_'+labels[cam]
                print 'Attempting to save', plotName
                plt.savefig(plotName)

            except:
                print 'FAILED'
        if booShow==True: plt.show()
        plt.close()        
        

# <codecell>

def RVCorr_RV(RVCorrMethod = 'PM', thisStarName = 'Giant01', sigmaClip = -1, RVClip = -1, booSave = False, booShow = True, booBaryPlot = False, booBaryCorrect = False, title = ''):
    
    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
    sigmas=np.load('npy/sigmas.npy')
    baryVels=np.load('npy/baryVels.npy')
    JDs=np.load('npy/JDs.npy')
    if RVCorrMethod=='PM':
        RVCorr=np.load('npy/RVCorr_PM.npy')
    else:
        RVCorr=np.load('npy/RVCorr_DM.npy')

    i = np.where(data[:,0]==thisStarName)[0][0]
#     X = range(data[:,0].shape[0])
#     Y = RVs
#     Y[Y==0.]=np.nan
    
#     if RVClip>-1:Y[np.abs(Y)>RVClip] = np.nan
#     if sigmaClip>-1:
#         stdY= np.std(Y)
#         medY = np.median(Y)
#         Y[(Y>=medY-sigmaClip*stdY) & (Y<=medY+sigmaClip*stdY)] = np.nan
    
    
    
    colors = ['b','g','r','cyan']
    labels = ['Blue','Green','Red','IR']
    
#     #Plots RVs, baryvels. all star, 4 cameras
#     print 'About to plot RVs from ',RVs.shape[0],'stars.'
    
    for cam in range(4)[:]:
        print data[i]
        print 'RV:',RVs[i,:,cam]
        print 'RVCorr:',RVCorr[i,:,cam]
        print 'JDs:',JDs
        RVs[np.abs(RVs)>RVClip] = np.nan
        plt.plot(JDs,RVs[i,:,cam], label = 'RV', marker='.')
        plt.plot(JDs,RVCorr[i,:,cam], label = 'Correction', marker='.')
        plt.plot(JDs,RVs[i,:,cam]-RVCorr[i,:,cam], label = 'Result', marker='.')

        if booBaryPlot==True: plt.plot(JDs, baryVels, label = 'Barycentric Vel. ', marker='.')

        plt.legend(loc=0)



#         ax.set_xticklabels(data[:,0])
#         ax.set_xticks(X)
#         plt.xticks(rotation=70)


        if title=='':
            plt.title('RVCorr and RV for '+data[i,0]+' - '+labels[cam]+' camera')
        else:
            plt.title(title)

#         Y[:,:,cam] = Y[:,:,cam][order[:,cam]]
#         YERR[:,:,cam] = YERR[:,:,cam][order[:,cam]]
        
#         #median
#         plt.scatter(X, stats.nanmedian(Y[:,:,cam], axis = 1), label = labels[cam], color = 'k')
        
#         #sigma
#         plt.scatter(X, stats.nanmedian(Y[:,:,cam], axis = 1)+stats.nanstd(Y[:,:,cam], axis = 1), label = labels[cam], color = 'r')
#         plt.scatter(X, stats.nanmedian(Y[:,:,cam], axis = 1)-stats.nanstd(Y[:,:,cam], axis = 1), label = labels[cam], color = 'r')
        
        
#         #min max
#         for star in range(Y.shape[0]):
#             x = np.nanmax(Y[star,:,cam])
#             n = np.nanmin(Y[star,:,cam])
#             plt.plot([star,star],[x,n], color = 'g', lw=2)
#             print star, n, x
        
#         plt.xlabel('MJD')
#         plt.ylabel('RV [m/s]')


        if booSave==True: 
            try:
                plotName = 'plots/RVCorr_RV_'+labels[cam]
                print 'Attempting to save', plotName
                plt.savefig(plotName)

            except:
                print 'FAILED'
        if booShow==True: plt.show()
        plt.close()        
        

# <codecell>

def SNR_RV_vs_fibre(RVClip = -1, booSave = False, booShow = True, title = ''):
    
    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
#     sigmas=np.load('npy/sigmas.npy')
#     baryVels=np.load('npy/baryVels.npy')
#     JDs=np.load('npy/JDs.npy')
    SNRs = np.load('npy/SNRs.npy')

    X = data[:,2].astype(float)

    labels = ['Blue','Green','Red','IR']
    
    #Plots RVs, baryvels. all star, 4 cameras
    print 'About to plot RVs and from ',X.shape[0],'fibres.'
    
    for cam in range(4)[:]:
        for epoch in range(RVs.shape[1])[:5]:
            fig, ax = plt.subplots()

    #         ax.set_xticklabels(data[:,0])
    #         ax.set_xticks(X)
    #         plt.xticks(rotation=70)

            if title=='':
                plt.title('RVs and SNR per fibre - t='+str(epoch)+' - '+labels[cam]+' camera')
            else:
                plt.title(title)
            
            ax2 = ax.twinx()

            Y1 = RVs[:,epoch,cam]    
            if RVClip>-1:Y1[np.abs(Y1)>RVClip] = np.nan

            Y2 = SNRs[:,epoch,cam]

            #RVs
            ax.scatter(X, Y1, label = 'RV', color = 'k')

            #SNRs
            ax2.scatter(X, Y2, label = 'SNR', color = 'r')

            plt.xlabel('Fibre #')
            plt.ylabel('RV [m/s]')
            ax.legend()
            ax2.legend()
    
            if booSave==True: 
                try:
                    plotName = 'plots/'+str(cam)+'/SNR_RV_FIBRE_'+str(epoch)
                    print 'Attempting to save', plotName
                    plt.savefig(plotName)

                except:
                    print 'FAILED'
            if booShow==True: plt.show()
            plt.close()        
            
            
        

# <codecell>

def RV_vs_fibre(RVClip = -1, booSave = False, booShow = True, title = ''):
    
    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
#     sigmas=np.load('npy/sigmas.npy')
#     baryVels=np.load('npy/baryVels.npy')
#     JDs=np.load('npy/JDs.npy')
    SNRs = np.load('npy/SNRs.npy')

    X = data[:,2].astype(float)

    labels = ['Blue','Green','Red','IR']
    
    #Plots RVs, baryvels. all star, 4 cameras
    print 'About to plot RVs and from ',X.shape[0],'fibres.'
    
    for cam in range(4)[:]:
        for epoch in range(RVs.shape[1])[:]:
            fig, ax = plt.subplots()

    #         ax.set_xticklabels(data[:,0])
    #         ax.set_xticks(X)
    #         plt.xticks(rotation=70)

            if title=='':
                plt.title('RVs vs fibre - t='+str(epoch)+' - '+labels[cam]+' camera')
            else:
                plt.title(title)
            
            Y1 = RVs[:,epoch,cam]    
            if RVClip>-1:Y1[np.abs(Y1)>RVClip] = np.nan

            #RVs
            ax.scatter(X, Y1, label = 'RV', color = 'k')

            plt.xlabel('Fibre #')
            plt.ylabel('RV [m/s]')
            ax.legend()
    
            if booSave==True: 
                try:
                    plotName = 'plots/'+str(cam+1)+'/RV_vs_FIBRE_'+str(epoch)
                    print 'Attempting to save', plotName
                    plt.savefig(plotName)

                except:
                    print 'FAILED'
            if booShow==True: plt.show()
            plt.close()        
            
            
        

# <codecell>

def SNR_vs_fibre(RVClip = -1, booSave = False, booShow = True, title = ''):
    
    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
#     sigmas=np.load('npy/sigmas.npy')
#     baryVels=np.load('npy/baryVels.npy')
#     JDs=np.load('npy/JDs.npy')
    SNRs = np.load('npy/SNRs.npy')

    X = data[:,2].astype(float)

    labels = ['Blue','Green','Red','IR']
    
    #Plots RVs, baryvels. all star, 4 cameras
    print 'About to plot RVs and from ',X.shape[0],'fibres.'
    
    for cam in range(4)[:]:
        for epoch in range(RVs.shape[1])[:]:
            fig, ax = plt.subplots()

    #         ax.set_xticklabels(data[:,0])
    #         ax.set_xticks(X)
    #         plt.xticks(rotation=70)

            if title=='':
                plt.title('SNR vs fibre - t='+str(epoch)+' - '+labels[cam]+' camera')
            else:
                plt.title(title)
            
            Y2 = SNRs[:,epoch,cam]

            #SNRs
            ax.scatter(X, Y2, label = 'SNR', color = 'r')

            plt.xlabel('Fibre #')
            plt.ylabel('RV [m/s]')
            ax.legend()
    
            if booSave==True: 
                try:
                    plotName = 'plots/'+str(cam+1)+'/SNR_vs_FIBRE_'+str(epoch)
                    print 'Attempting to save', plotName
                    plt.savefig(plotName)

                except:
                    print 'FAILED'
            if booShow==True: plt.show()
            plt.close()        
            
            
        

# <codecell>

def flux_and_CC(RVref=5000, booSave = False, booShow = True):
    #Creates plots of 2fluxes and the corresponding CC for RVs >RVref
    import RVTools

    labels = ['Blue','Green','Red','IR']
    
    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
    i=0
    for cam in range(RVs.shape[2])[1:2]:
        for epoch in range(RVs.shape[1])[3:]:
            for star in range(RVs.shape[0])[:]:
#                 if np.abs(RVs[star,epoch,cam])>RVref:
#                 if ((RVs[star,epoch,cam]<-1000) & (RVs[star,epoch,cam]>-3000)):
                i+=1
                print i, RVs[star,epoch,cam],data[star]
                fileName = 'red_'+data[star,0]+'.obj'
                print 'Ploting fluxes from', fileName 
                filehandler = open('obj/'+fileName, 'r')
                thisStar = pickle.load(filehandler)

                lambda1,flux1, lambda2,flux2, CCCurve, p, x_mask, RV = RVTools.single_RVs_CC_t0(thisStar, cam = cam, t = epoch)

                plt.subplot(2,1,1)
                plt.title(thisStar.name +' '+ labels[cam]+' Original Fluxes - RV='+str(RV))
                plt.plot(lambda1,flux1, label='t0')
                plt.plot(lambda2,flux2, label='t'+str(epoch))
                plt.legend(loc=0)
                plt.subplot(2,1,2)
                plt.title('Cross Correlation')
                plt.plot(CCCurve/np.max(CCCurve), label='CC Curve')
                plt.plot(x_mask,RVTools.gaussian(x_mask, p[0], p[1] ), label='Gaussian fit', c='r')
                plt.legend(loc=0)        

                if booSave==True: 
                    try:
                        plotName = 'plots/'+str(cam+1)+'/CC_'+data[star,0]+'_'+str(epoch)+'_'+str(cam)
                        print 'Attempting to save', plotName
                        plt.savefig(plotName)

                    except:
                        print 'FAILED'
                if booShow==True: plt.show()
                plt.close()        


#                     if input('continue?')!=1:
#                         break


                thisStar = None
                filehandler.close()
                print 

