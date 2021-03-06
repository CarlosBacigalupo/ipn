
# coding: utf-8

# In[2]:

import pylab as plt
import numpy as np
import glob
import pickle
import scipy.stats as stats
import RVTools as RVT


# In[29]:

labels = ['Blue','Green','Red','IR']

colors = ['b','g','r','c']

cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}


# In[22]:

import numpy as np
a = np.array([1,2,3,4,4,4,4,4,4])
np.array(cnames.values())[a]


# In[ ]:

def all_spec_overlap(thisStar, thisCamIdx = '', booShow = True, booSave = False):
    import gc
    '''
    All spectra for thisCam, or all cams if thisCam is skipped.
    '''
    gc.enable()
    
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
        ax[0,0].set_ylim(-50 ,(nFluxes+2)*10)
    
        for i in np.arange(nFluxes):
                if np.sum(thisCam.wavelengths[i])>0:
                    d, f = thisCam.wavelengths[i], thisCam.red_fluxes[i]/np.abs(np.median(thisCam.red_fluxes[i]))
                    if cam ==0:
                        ax[0,0].plot(d, f+i*10, 'b')
                    elif cam==1:
                        ax[0,1].plot(d, f+i*10, 'g')
                    elif cam==2:
                        ax[1,0].plot(d, f+i*10, 'r')
                    elif cam==3:
                        ax[1,1].plot(d, f+i*10, 'cyan')
                #         ax.plot(d, f+thisStar.exposures.JDs[i], 'k')
    thisCam = None
    name = thisStar.name
    thisStar = None
    fig = None
    ax = None
    time = None
    gc.collect()
    plt.xlabel('Wavelength [Ang]')
    plt.title(name+' - Camera '+str(cam+1))
#     plt.gca().set_yticks(range(thisCam.wavelengths.shape[0]))
#     plt.gca().set_yticklabels(thisCam.fileNames)

#     ax.set_yticklabels(thisCam.fileNames)
#     ax.set_yticks(range(thisCam.wavelengths.shape[0]))

    
#     plt.xticks(rotation=70)

#     ax[0,0].set_yticklabels(fileNames)
    if booSave==True: 
#         try:
        plt.savefig('plots/' + name + 'SOvl_cam' + str(cam+1), dpi = 1000 )
#         except:
#             print 'Couldn''t save'
    if booShow==True: plt.show()

    plt.close()
    


# In[ ]:

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


# In[ ]:

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
                
        


# In[1]:

def RVs_all_stars_NPYs(booShowArcRVs = False, idStars = [], sigmaClip = -1, RVClip = -1, topStars = -1, booSave = False, booShow = True, booBaryPlot = False, booBaryCorrect = False, title = '', legend = True):
    
    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
    sigmas=np.load('npy/sigmas.npy')
    baryVels=np.load('npy/baryVels.npy')
    JDs=np.load('npy/JDs.npy')
    try:
        arcRVS=np.load('npy/arcRVs.npy')
    except:
        print 'Not including arcRVs'
        booShowArcRVs = False
    
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
                plt.title(title +' - '+labels[cam]+' camera')
            
            if data[i,0] in idStars:
                m = '*'
                s=100
                c='k'
            else:
                m = 'o'
                s=5
                c=colors[cam]
                
            
#             plt.errorbar(X, Y[i,:,cam], yerr=YERR[i,:,cam]*1000, fmt='.', label = labels[cam], color = colors[cam])
            if booBaryCorrect==True: 
                plt.scatter(X, Y[i,:,cam]- baryVels, color = c, s=s, marker=m, label = 'Stars')

            else:
                plt.scatter(X, Y[i,:,cam], color = c, s=s, marker=m, label = 'Stars')

            if booShowArcRVs==True:
                if data[i,0] in idStars:
                    print i,data[i,:],RVT.pivot2idx(data[i,2].astype(int))
                    arcRVs = np.load('npy/arcRVs.npy')
                    plt.scatter(X, arcRVs[RVT.pivot2idx(data[i,2].astype(int)),:,cam], marker = '+' , label = 'ARC RV', color = 'm', s=500)
                    plt.scatter(X, np.nanmean(arcRVs, axis=0)[:,cam], marker = '_' , label = 'mean ARC RV', color='k', s=300)
                
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
        


        plt.grid(True)
        if legend==True: plt.legend(loc=0)
        if booSave==True: 
            try:
                plotName = 'plots/All_RVs_'+labels[cam]+''
                print 'Attempting to save', plotName
                plt.savefig(plotName)

            except:
                print 'FAILED'
        if booShow==True: plt.show()
        plt.close()
                
        


# In[37]:

def RV_Tree(zoom = 1, sigmaClip = -1, RVClip = -1, booSave = False, booShow = True, booBaryPlot = False, booBaryCorrect = False, title = ''):
    
    data=np.load('npy/data.npy')
#     RVs=np.load('npy/RVs.npy')
    baryRVs=np.load('npy/baryRVs.npy')
#     sigmas=np.load('npy/sigmas.npy')
#     baryVels=np.load('npy/baryVels.npy')
#     JDs=np.load('npy/JDs.npy')
#     baryVels3D = np.zeros(RVs.shape)
#     baryVels3D[:,:,0] = np.tile(baryVels,[RVs.shape[0],1])
#     baryVels3D[:,:,1] = np.tile(baryVels,[RVs.shape[0],1])
#     baryVels3D[:,:,2] = np.tile(baryVels,[RVs.shape[0],1])
#     baryVels3D[:,:,3] = np.tile(baryVels,[RVs.shape[0],1])
    
    
    #Plots RVs, baryvels. all star, 4 cameras
    print 'About to plot RVs from ',baryRVs.shape[0],'stars.'

#     baryRVs = RVs - baryVels3D
    
#     if RVClip>-1:Y[np.abs(Y)>RVClip] = np.nan
#     if sigmaClip>-1:
#         stdY= np.std(Y)
#         medY = np.median(Y)
#         Y[(Y>=medY-sigmaClip*stdY) & (Y<=medY+sigmaClip*stdY)] = np.nan
    
#     YERR = sigmas

    X = np.arange(data[:,0].shape[0])
#     bcRVs = RVs-baryVels
    
#     Y = RVs
#     Y[Y==0.]=np.nan
#     order = np.argsort(np.nanstd(RVs,axis=1),axis=0)    
#     order = np.argsort(np.ptp(np.nan_to_num(baryRVs),1),0)
    order = np.argsort(np.nanmax(baryRVs,1)-np.nanmin(baryRVs,1),0)
    
    if zoom > 1 :
        zoomList = [np.arange(len(X))]
    
    steps = np.zeros(zoom+1)
    for i in range(zoom+1):
        steps[i] = len(X)/zoom*i
        
    for i in range(zoom):
        print 'i.zoom',i,zoom
        if i<(zoom-1):
            zoomList.append([np.arange(steps[i],steps[i+1]).astype(int)])
        else:
            print 'last zoom'
            zoomList.append([np.arange(steps[i],len(X)).astype(int)])
    
    zoomList = np.array(zoomList)
    
    print 'steps,zoomList',steps,zoomList
    for cam in range(4)[:]:
        for i, thisXRange in enumerate(zoomList):
            
            print 'thisXRange',thisXRange
            thisOrder = order[:,cam][thisXRange]
            print 'thisOrder',thisOrder
            
            fig, ax = plt.subplots()
            ax.set_xticklabels(data[:,0][thisOrder])
            ax.set_xticks(X[range(len(thisOrder))])
            plt.xticks(rotation=90)

#             if booBaryPlot==True: plt.plot(X, baryVels, label = 'Barycentric Vel. ')


            thisY = baryRVs[:,:,cam][thisOrder]
            thisY[thisY==0.]=np.nan

#             YERR[:,:,cam] = YERR[:,:,cam][order[:,cam]]

            #median
            plt.scatter(X[range(len(thisOrder))], stats.nanmedian(thisY , axis = 1), label = labels[cam], color = 'k')

            #sigma
            plt.scatter(X[range(len(thisOrder))], stats.nanmedian(thisY, axis = 1)+stats.nanstd(thisY, axis = 1), label = labels[cam], color = 'r')
            plt.scatter(X[range(len(thisOrder))], stats.nanmedian(thisY, axis = 1)-stats.nanstd(thisY, axis = 1), label = labels[cam], color = 'r')


            #min max
            for star in range(thisY.shape[0]):
                x = np.nanmax(thisY[star,:])
                n = np.nanmin(thisY[star,:])
                plt.plot([star,star],[x,n], color = 'g', lw=2)
                print star, n, x, x-n

            #zero
            plt.plot(X[range(len(thisOrder))],np.zeros(len(thisOrder)), '--')

            plt.grid(axis='x')
            plt.xlabel('Stars')
            plt.ylabel('RV [m/s]')
        #     plt.legend(loc=0)

            fig.tight_layout()
            if booSave==True: 
                try:
                    plotName = 'plots/tree/'+str(cam+1)+'/Tree_'+str(i)+'_'+labels[cam]
                    print 'Attempting to save', plotName
                    plt.savefig(plotName)

                except:
                    print 'FAILED'
            if booShow==True: plt.show()
            plt.close()        
        


# In[ ]:

def RV_Tree2(zoom = 1, sigmaClip = -1, RVClip = -1, booSave = False, booShow = True, booBaryPlot = False, booBaryCorrect = False, title = ''):
    
    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
#     sigmas=np.load('npy/sigmas.npy')
    baryVels=np.load('npy/baryVels.npy')
#     JDs=np.load('npy/JDs.npy')
    baryVels3D = np.zeros(RVs.shape)
    baryVels3D[:,:,0] = np.tile(baryVels,[RVs.shape[0],1])
    baryVels3D[:,:,1] = np.tile(baryVels,[RVs.shape[0],1])
    baryVels3D[:,:,2] = np.tile(baryVels,[RVs.shape[0],1])
    baryVels3D[:,:,3] = np.tile(baryVels,[RVs.shape[0],1])
    
    
    #Plots RVs, baryvels. all star, 4 cameras
    print 'About to plot RVs from ',RVs.shape[0],'stars.'

    baryRVs = RVs - baryVels3D
    
#     if RVClip>-1:Y[np.abs(Y)>RVClip] = np.nan
#     if sigmaClip>-1:
#         stdY= np.std(Y)
#         medY = np.median(Y)
#         Y[(Y>=medY-sigmaClip*stdY) & (Y<=medY+sigmaClip*stdY)] = np.nan
    
#     YERR = sigmas

    X = np.arange(data[:,0].shape[0])
#     bcRVs = RVs-baryVels
    
#     Y = RVs
#     Y[Y==0.]=np.nan
#     order = np.argsort(np.nanstd(RVs,axis=1),axis=0)    
    order = np.argsort(np.ptp(baryVels3D,1),0)
    
    if zoom > 1 :
        zoomList = [np.arange(len(X))]
    
    steps = np.zeros(zoom+1)
    for i in range(zoom+1):
        steps[i] = len(X)/zoom*i
        
    for i in range(zoom):
        print 'i.zoom',i,zoom
        if i<(zoom-1):
            zoomList.append([np.arange(steps[i],steps[i+1]).astype(int)])
        else:
            print 'last zoom'
            zoomList.append([np.arange(steps[i],len(X)).astype(int)])
    
    zoomList = np.array(zoomList)
    
    print 'steps,zoomList',steps,zoomList
    for cam in range(4)[:1]:
        for i, thisXRange in enumerate(zoomList):
            
            print 'thisXRange',thisXRange
            thisOrder = order[:,cam][thisXRange]
            print 'thisOrder',thisOrder
            
            fig, ax = plt.subplots()
            ax.set_xticklabels(data[:,0][thisOrder])
            ax.set_xticks(X[range(len(thisOrder))])
            plt.xticks(rotation=90)

#             if booBaryPlot==True: plt.plot(X, baryVels, label = 'Barycentric Vel. ')


            thisY = baryRVs[:,:,cam][thisOrder]
            thisY[thisY==0.]=np.nan

#             YERR[:,:,cam] = YERR[:,:,cam][order[:,cam]]

            #median
            plt.scatter(X[range(len(thisOrder))], stats.nanmedian(thisY , axis = 1), label = labels[cam], color = 'k')

            #sigma
            plt.scatter(X[range(len(thisOrder))], stats.nanmedian(thisY, axis = 1)+stats.nanstd(thisY, axis = 1), label = labels[cam], color = 'r')
            plt.scatter(X[range(len(thisOrder))], stats.nanmedian(thisY, axis = 1)-stats.nanstd(thisY, axis = 1), label = labels[cam], color = 'r')


            #min max
            for star in range(thisY.shape[0]):
                x = np.nanmax(thisY[star,:])
                n = np.nanmin(thisY[star,:])
                plt.plot([star,star],[x,n], color = 'g', lw=2)
                print star, n, x, x-n

            #zero
            plt.plot(X[range(len(thisOrder))],np.zeros(len(thisOrder)), '--')

            plt.grid(axis='x')
            plt.xlabel('Stars')
            plt.ylabel('RV [m/s]')
        #     plt.legend(loc=0)

            fig.tight_layout()
            if booSave==True: 
                try:
                    plotName = 'plots/Tree_'+str(i)+'_'+labels[cam]
                    print 'Attempting to save', plotName
                    plt.savefig(plotName)

                except:
                    print 'FAILED'
            if booShow==True: plt.show()
            plt.close()        
        


# In[ ]:

def sineFit(booSave = False, booShow = True, starIdx=-1, booShowAvg = True):

    data = np.load('npy/data.npy')
    sineFit = np.load('npy/sineFit.npy')
    baryRVs = np.load('npy/baryRVs.npy')
    MJDs = np.load('npy/MJDs.npy')
    avgSineFit = np.load('npy/avgSineFit.npy')
    avgBaryRVs = np.load('npy/avgBaryRVs.npy')
    avgMJDs = np.load('npy/avgMJDs.npy')


    for cam in range(4):
        for i,thisSineFit in enumerate(sineFit[:,:,cam]):
            if ((starIdx==-1) or (starIdx==i)):
                thisBaryRVs = baryRVs[i,:,cam]
                thisMJDs = np.linspace(np.min(MJDs),np.max(MJDs), num=100)

                sineCurve = thisSineFit[0]*np.sin(2*np.pi/thisSineFit[2]*(thisMJDs+thisSineFit[1]))

                plt.plot(MJDs,thisBaryRVs, '.')
                thisLabel = 'All'+' RV:'+str("%.2f" % thisSineFit[0])
#                 thisLabel += ' ph:'+str("%.2f" % thisSineFit[1])
                thisLabel += ' P:'+str("%.2f" % thisSineFit[2])
                thisLabel += ' std:'+str("%.2f" % thisSineFit[3])
                plt.plot(thisMJDs,sineCurve, label = thisLabel)
                
                if booShowAvg==True:
                    thisAvgBaryRVs = avgBaryRVs[i,:,cam]
#                     thisMJDs = np.linspace(np.min(MJDs),np.max(MJDs))

                    avgSineCurve = avgSineFit[i,0,cam]*np.sin(2*np.pi/avgSineFit[i,2,cam]*(thisMJDs+avgSineFit[i,1,cam]))

                    plt.plot(avgMJDs,thisAvgBaryRVs, '.')
                    thisLabel = 'Avg'+' RV:'+str("%.2f" % avgSineFit[i,0,cam])
#                     thisLabel += ' ph:'+str("%.2f" % avgSineFit[i,1,cam])
                    thisLabel += ' P:'+str("%.2f" % avgSineFit[i,2,cam])
                    thisLabel += ' std:'+str("%.2f" % avgSineFit[i,3,cam])                    
                    plt.plot(thisMJDs,avgSineCurve, label = thisLabel)
                    
                    
                    
                plt.title(data[i,0])
                plt.legend(loc=0,prop={'size':6})

                if booSave==True: 
    #                 try:
                    name = data[i,0].replace('.','_')
                    plotName = 'plots/sine/'+str(cam+1)+'/Sine_'+name+'_'+labels[cam]
                    print 'Attempting to save', plotName
                    plt.savefig(plotName)

    #                 except:
    #                     print 'FAILED'
                if booShow==True: plt.show()
                plt.close()        


# In[1]:

def SNR_W(RVCorrMethod = 'PM', thisStarName = 'Giant01', sigmaClip = -1, RVClip = -1, booSave = False, booShow = True, booBaryPlot = False, booBaryCorrect = False, title = ''):
#plots SNR and W vs value

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
        


# In[ ]:

def RVCorr_RV(thisStarName = 'Giant01', RVClip = -1, booSave = False, booShow = True, booBaryPlot = False, booBaryCorrect = False, title = ''):
    
    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
    sigmas=np.load('npy/sigmas.npy')
    baryVels=np.load('npy/baryVels.npy')
    JDs=np.load('npy/JDs.npy')    
    RVCorr_PM=np.load('npy/RVCorr_PM.npy')
    RVCorr_DM=np.load('npy/RVCorr_DM.npy')
    cRVs_PMDM=np.load('npy/cRVs_PMDM.npy')

    idx = np.where(data[:,0]==thisStarName)[0][0]
    
    methods = ['PM', 'DM', 'PMDM']
    
    for i,RVCorr in enumerate([RVCorr_PM, RVCorr_DM,'']):
        for cam in range(4)[:]:
#             print data[idx]
#             print 'RV:',RVs[idx,:,cam]
#             print 'RVCorr:',RVCorr[idx,:,cam]
#             print 'JDs:',JDs
            RVs[np.abs(RVs)>RVClip] = np.nan

            plt.plot(JDs,RVs[idx,:,cam], label = 'RV', marker='.', c='b')
            if booBaryPlot==True: plt.plot(JDs, baryVels, label = 'Barycentric Vel. ', marker='.', c='cyan')
        
            if i==0:
                plt.plot(JDs,RVCorr[idx,:,cam], label = 'Correction', marker='.', c='r')
                plt.plot(JDs,RVs[idx,:,cam]-RVCorr[idx,:,cam], label = 'Result', marker='.', c='g')
            elif i==1:
                plt.plot(JDs,RVCorr[idx,:,cam], label = 'Correction', marker='.', c='r')
                plt.plot(JDs,RVs[idx,:,cam]-RVCorr[idx,:,cam], label = 'Result', marker='.', c='g')
            elif i==2:
                plt.plot(JDs,cRVs_PMDM[idx,:,cam], label = 'Result', marker='.', c='g')

            plt.legend(loc=0)
            plt.title('RVCorr_'+methods[i]+' for '+data[idx,0]+' - '+labels[cam]+' camera')

            if booSave==True: 
                try:
                    plotName = 'plots/RVCorr_'+methods[i]+'_'+labels[cam]
                    print 'Attempting to save', plotName
                    plt.savefig(plotName)

                except:
                    print 'FAILED'
            if booShow==True: plt.show()
            plt.close()        


# In[2]:

def RVCorr_Slit(thisStarName = 'Giant01', RVClip = -1, booSave = False, booShow = True, booBaryPlot = False, booBaryCorrect = False, title = ''):
    
    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
    sigmas=np.load('npy/sigmas.npy')
    baryVels=np.load('npy/baryVels.npy')
    JDs=np.load('npy/JDs.npy')    
    RVCorr_PM=np.load('npy/RVCorr_PM.npy')
    RVCorr_DM=np.load('npy/RVCorr_DM.npy')
    allW_PM=np.load('npy/allW_PM.npy')
    allW_DM=np.load('npy/allW_DM.npy')
    cRVs_PMDM=np.load('npy/cRVs_PMDM.npy')

    #load function that translates pivot# to y-pixel  p2y(pivot)=y-pixel of pivot
    p2y = RVT.pivot_to_y('/Users/Carlos/Documents/HERMES/reductions/6.2/rhoTuc_6.2/0_20aug/1/20aug10042tlm.fits') 

    #gets the y position of for the data array
    datay = p2y[data[:,2].astype(float).astype(int)]
    
    idx = np.where(data[:,0]==thisStarName)[0][0]
   
    methods = ['PM', 'DM', 'PMDM']
    i=0
#     for i,RVCorr in enumerate([RVCorr_PM, RVCorr_DM,'']):
    
    for cam in range(4)[:1]:
#         RVs[np.abs(RVs)>RVClip] = np.nan
        plt.scatter(allW_PM[:,cam,idx],datay, s=2, c='k')
        plt.plot([0,0],[np.min(datay),np.max(datay)])
        plt.plot([0,0],[np.min(datay),np.max(datay)])
        plt.xlim(-np.max(allW_PM[:,cam,idx]),np.max(allW_PM[:,cam,idx]))
        plt.plot(plt.xlim(),[datay[idx],datay[idx]])
        
        
#         plt.plot(JDs,RVs[idx,:,cam], label = 'RV', marker='.', c='b')
#         if booBaryPlot==True: plt.plot(JDs, baryVels, label = 'Barycentric Vel. ', marker='.', c='cyan')

#         if i==0:
#             plt.plot(JDs,RVCorr[idx,:,cam], label = 'Correction', marker='.', c='r')
#             plt.plot(JDs,RVs[idx,:,cam]-RVCorr[idx,:,cam], label = 'Result', marker='.', c='g')
#         elif i==1:
#             plt.plot(JDs,RVCorr[idx,:,cam], label = 'Correction', marker='.', c='r')
#             plt.plot(JDs,RVs[idx,:,cam]-RVCorr[idx,:,cam], label = 'Result', marker='.', c='g')
#         elif i==2:
#             plt.plot(JDs,cRVs_PMDM[idx,:,cam], label = 'Result', marker='.', c='g')

#         plt.legend(loc=0)
        plt.title('RVCorr_'+methods[i]+' for '+data[idx,0]+' - '+labels[cam]+' camera')

        if booSave==True: 
            try:
                plotName = 'plots/RVCorr_'+methods[i]+'_'+labels[cam]
                print 'Attempting to save', plotName
                plt.savefig(plotName)

            except:
                print 'FAILED'
        if booShow==True: plt.show()
        plt.close()        


# In[38]:

def SNR_RV_vs_fibre(RVClip = -1, booSave = False, booShow = True, title = ''):
    
    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
#     sigmas=np.load('npy/sigmas.npy')
#     baryVels=np.load('npy/baryVels.npy')
#     JDs=np.load('npy/JDs.npy')
    SNRs = np.load('npy/SNRs.npy')

    X = data[:,2].astype(float)

    
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
            
            
        


# In[39]:

def RV_vs_fibre(RVClip = -1, booSave = False, booShow = True, title = ''):
    
    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
#     sigmas=np.load('npy/sigmas.npy')
#     baryVels=np.load('npy/baryVels.npy')
#     JDs=np.load('npy/JDs.npy')
    SNRs = np.load('npy/SNRs.npy')

    X = data[:,2].astype(float)

    
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
            
            
        


# In[40]:

def SNR_vs_fibre(RVClip = -1, booSave = False, booShow = True, title = ''):
    
    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
#     sigmas=np.load('npy/sigmas.npy')
#     baryVels=np.load('npy/baryVels.npy')
#     JDs=np.load('npy/JDs.npy')
    SNRs = np.load('npy/SNRs.npy')

    X = data[:,2].astype(float)

    
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
            
            
        


# In[41]:

def flux_and_CC(RVref=5000, booSave = False, booShow = True):
    #Creates plots of 2fluxes and the corresponding CC for RVs >RVref

    
    data=np.load('npy/data.npy')
    RVs=np.load('npy/RVs.npy')
    i=0
    for cam in range(RVs.shape[2])[:]:
        for epoch in range(RVs.shape[1])[:]:
            for star in range(RVs.shape[0])[:]:
#                 if np.abs(RVs[star,epoch,cam])>RVref:
#                 if ((RVs[star,epoch,cam]<-1000) & (RVs[star,epoch,cam]>-3000)):
                i+=1
                print i, RVs[star,epoch,cam],data[star]
                fileName = 'red_'+data[star,0]+'.obj'
                print 'Ploting fluxes from', fileName 
                filehandler = open('obj/'+fileName, 'r')
                thisStar = pickle.load(filehandler)

                lambda1,flux1, lambda2,flux2, CCCurve, p, x_mask, RV = RVT.single_RVs_CC_t0(thisStar, cam = cam, t = epoch)

                plt.subplot(2,1,1)
                plt.title(thisStar.name +' '+ labels[cam]+' Original Fluxes - RV='+str(RV))
                plt.plot(lambda1,flux1, label='t0')
                plt.plot(lambda2,flux2, label='t'+str(epoch))
                plt.legend(loc=0)
                plt.subplot(2,1,2)
                plt.title('Cross Correlation')
                plt.plot(CCCurve/np.max(CCCurve), label='CC Curve')
                plt.plot(x_mask,RVT.gaussian(x_mask, p[0], p[1] ), label='Gaussian fit', c='r')
                plt.legend(loc=0)        

                if booSave==True: 
                    try:
                        plotName = 'plots/'+str(cam+1)+'/CC_'+data[star,0]+'_'+str(epoch)+'_'+str(cam+1)
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


# In[ ]:

def arcRVs(booSave = False, booShow = True, booFit = False):
    arcRVs = np.load('npy/arcRVs.npy')
    MJDs = np.load('npy/JDs.npy')

    for epoch,MJD in enumerate(MJDs):
        for cam in range(4):
            y = arcRVs[:,epoch,cam]
            plt.plot(y,'.'+colors[cam])
            if booFit==True:
                x = np.arange(len(y))
                p = np.polyfit(x[-np.isnan(y)],y[-np.isnan(y)],1)
                plt.plot(x,x*p[0]+p[1])
                
        plt.title('MJD='+str(MJD))
        if booSave==True: 
            try:
                plotName = 'plots/arcRVs_'+str(epoch)
                print 'Attempting to save', plotName
                plt.savefig(plotName)

            except Exception,e: 
                print str(e)
                print 'FAILED'
        if booShow==True: plt.show()
        plt.close()        

    
    


# In[ ]:

def RVAvgGroups(booSave = False, booShow = True):
    baryRVs = np.load('npy/baryRVs.npy')
    MJDs = np.load('npy/MJDs.npy')
    avgIdxGroups = np.load('npy/avgIdxGroups.npy')

    c = np.array(colors)[avgIdxGroups.astype(int) % len(colors)]

    for cam in range(4):    
        plt.scatter(MJDs, baryRVs[1,:,cam], c = c)
                
        plt.title('Average Groups')
        if booSave==True: 
            try:
                plotName = 'plots/RVAvgGroups/RVAvgGroups_'+labels[cam]
                print 'Attempting to save', plotName
                plt.savefig(plotName)

            except Exception,e: 
                print str(e)
                print 'FAILED'
        if booShow==True: plt.show()
        plt.close()        

