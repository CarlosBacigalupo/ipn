# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# from os import walk, system
# #import _mysql
# import psycopg2 as mdb
import pyfits as pf
# import time
# import numpy as np
# import ephem
from pyraf import iraf
import glob
# import fileinput
# from matplotlib import *
# from pylab import *
# from mpl_toolkits.mplot3d import Axes3D
# import itertools
# import sys
# import galahParameters as gp

# <codecell>

def shift_master(ref, master):
    """
    Find the shift between two images.
    returns the shift in y direction (along columns)
    """
    #make a narow copy of the reference flat and masterflat 
    iraf.imcopy(input="../"+ref+"[1990:2010,*]", output="tmp/masterflat_ref.fitscut",Stdout="/dev/null")
    iraf.imcopy(input="tmp/masterflat.fits[1990:2010,*]", output="tmp/masterflat.fitscut",Stdout="/dev/null")

    #find the shift with correlation
    iraf.images(_doprint=0,Stdout="/dev/null")
    iraf.immatch(_doprint=0,Stdout="/dev/null")

    shift=iraf.xregister(input="tmp/masterflat_ref.fitscut, tmp/masterflat.fitscut", reference="tmp/masterflat_ref.fitscut", regions='[*,*]', shifts="tmp/mastershift", output="", databasefmt="no", correlation="fourier", xwindow=3, ywindow=51, xcbox=21, ycbox=21, Stdout=1)

    shift=float(shift[-2].split()[-2])

    #Shift the list of apertures
    f=open(master, "r")
    o=open("database/aptmp_masterflat_s", "w")

    for line in f:
        l=line.split()
        if len(l)>0 and l[0]=='begin':
            l[-1]=str(float(l[-1])-shift)
            o.write(l[0]+"\t"+l[1]+"\t"+"tmp/masterflat_s"+"\t"+l[3]+"\t"+l[4]+"\t"+l[5]+"\n")
        elif len(l)>0 and l[0]=='center':
            l[-1]=str(float(l[-1])-shift)
            o.write("\t"+l[0]+"\t"+l[1]+"\t"+l[2]+"\n")
        elif len(l)>0 and l[0]=='image':
            o.write("\t"+"image"+"\t"+"tmp/masterflat_s"+"\n")
        else:
            o.write(line)

    f.close()
    o.close()

# <codecell>

def extract(arcs, objs,ccd,mode=1):
    """
    mode=1: 1D extraction
    mode=2: 2D extraction
    """
    #apall na vseh luckah
    print " + identifing arcs"

    for i in arcs:
        iraf.hedit(images='tmp/arcs/'+i[1], fields='DISPAXIS', value=1, add='yes', verify='no', update='yes',Stdout="/dev/null")
        iraf.apall(input='tmp/arcs/%s' % (i[1]), referen='tmp/masterflat.fits', format='multispec', interac='no', find='no', recenter='no', resize='no', edit='no', trace='no', fittrac='no', extract='yes', extras='no', review='yes', lower=-3.0, upper=3.0, nsubaps=7, pfit="fit1d", Stdout="/dev/null")

    if mode==1:
        pass
    else:
        iraf.cd("tmp/arcs")
        geometry_prep(arcs,ccd)
        os.system("cp transformations* ../objs/")
        #sys.exit(0)
        iraf.cd("../..")
        for i in arcs:
            os.system("rm tmp/arcs/%s.ms.fits" % (i[1][:-5]))
            os.system("rm tmp/arcs/%s_cut*.fits" % (i[1][:-5]))
            os.system("rm tmp/arcs/%s_t*.fits" % (i[1][:-5]))
            pass

        os.system("rm tmp/arcs/calibrations/idarcs_cut*")
        os.system("rm tmp/arcs/arcs_cut*")

        #extract 2d arcs and objects
        for i in arcs:
            iraf.hedit(images='tmp/arcs/'+i[1], fields='DISPAXIS', value=1, add='yes', verify='no', update='yes',Stdout="/dev/null")
            iraf.apall(input='tmp/arcs/%s' % (i[1]), referen='tmp/masterflat.fits', format='multispec', interac='no', find='no', recenter='no', resize='no', edit='no', trace='no', fittrac='no', extract='yes', extras='no', review='yes', lower=-3.0, upper=3.0, nsubaps=7, pfit="fit1d", Stdout="/dev/null")
            #forget apertures
            for j in range(8,1000):
                iraf.hedit(images='tmp/arcs/%s.ms.fits' % (i[1][:-5]), fields="APNUM%s" % (str(j)), value='', delete="yes", verify="no", Stdout="/dev/null")

        for i in objs:
            iraf.hedit(images='tmp/objs/'+i[1], fields='DISPAXIS', value=1, add='yes', verify='no', update='yes',Stdout="/dev/null")
            iraf.apall(input='tmp/objs/%s' % (i[1]), referen='tmp/masterflat.fits', format='multispec', interac='no', find='no', recenter='no', resize='no', edit='no', trace='no', fittrac='no', extract='yes', extras='no', review='yes', lower=-3.0, upper=3.0, nsubaps=7, pfit="fit1d", Stdout="/dev/null")
            #forget apertures
            for j in range(8,1000):
                iraf.hedit(images='tmp/objs/%s.ms.fits' % (i[1][:-5]), fields="APNUM%s" % (str(j)), value='', delete="yes", verify="no", Stdout="/dev/null")

        iraf.cd("tmp/arcs")
        for ii in arcs:
            geometry_transform(ii)
        iraf.cd("../..")

        iraf.cd("tmp/objs")
        for ii in objs:
            geometry_transform(ii)
        iraf.cd("../..")

    #make normal 1d extraction and copy results into it
    for i in arcs:
        os.system("rm -f tmp/arcs/%s" % (i[1][:-5]+".ms.fits"))
        iraf.apall(input='tmp/arcs/%s' % (i[1]), referen='tmp/masterflat.fits', format='multispec', interac='no', find='no', recenter='no', resize='no', edit='no', trace='no', fittrac='no', extract='yes', extras='no', review='yes', lower=-3.0, upper=3.0, nsubaps=1, pfit="fit1d", Stdout="/dev/null")
        os.system("cp tmp/arcs/%s" % (i[1][:-5])+".ms.fits tmp/arcs/%s" % (i[1][:-5])+".ms2.fits")

        if mode==1:
            pass
        else:
            for j in range(1,393):
                os.system("rm -f tmp/copy_tmp.fits")
                try:
                    iraf.blkavg(input="tmp/arcs/"+i[1][:-5]+"_t%s.fits" % (str(j)), output="tmp/copy_tmp", option='sum', b1=1, b2=7, Stdout="/dev/null")
                    iraf.imcopy(input="tmp/copy_tmp", output="tmp/arcs/"+i[1][:-5]+".ms.fits[*,%s]" % (j), Stdout="/dev/null")
                except:
                    pass

    for i in objs:
        os.system("rm -f tmp/objs/%s" % (i[1][:-5]+".ms.fits"))
        iraf.apall(input='tmp/objs/%s' % (i[1]), referen='tmp/masterflat.fits', format='multispec', interac='no', find='no', recenter='no', resize='no', edit='no', trace='no', fittrac='no', extract='yes', extras='no', review='yes', lower=-3.0, upper=3.0, nsubaps=1, pfit="fit1d", Stdout="/dev/null")
        os.system("cp tmp/objs/%s" % (i[1][:-5])+".ms.fits tmp/objs/%s" % (i[1][:-5])+".ms2.fits")

        if mode==1:
            pass
        else:
            for j in range(1,393):
                os.system("rm -f tmp/copy_tmp.fits")
                try:
                    iraf.blkavg(input="tmp/objs/"+i[1][:-5]+"_t%s.fits" % (str(j)), output="tmp/copy_tmp", option='sum', b1=1, b2=7, Stdout="/dev/null")
                    iraf.imcopy(input="tmp/copy_tmp", output="tmp/objs/"+i[1][:-5]+".ms.fits[*,%s]" % (j), Stdout="/dev/null")
                except:
                    pass

# <headingcell level=3>

# BIAS

# <codecell>

cd ~/Documents/HERMES/reductions/iraf/bias/140821/1

# <codecell>

biases = glob.glob('*.fits')`

# <codecell>

biases = ",".join(biases)

# <codecell>

iraf.zerocombine(input=biases, output='masterbias', combine='median', reject='none', ccdtype='',Stdout="/dev/null")

# <codecell>

cd ~/Documents/HERMES/reductions/iraf/HD1581/0_20aug/1/

# <codecell>

a = pf.open('20aug10052.fits')
a[0].header['RUNCMD']

# <headingcell level=3>

# FLAT

# <codecell>

flat = '20aug10034.fits'
masterflat = 'masterflat_blue0.fits'

# <codecell>

iraf.noao(_doprint=0,Stdout="/dev/null")
iraf.imred(_doprint=0,Stdout="/dev/null")
iraf.ccdred(_doprint=0,Stdout="/dev/null")

# <codecell>

#bias subtract
iraf.ccdproc(images=flat, ccdtype='', fixpix='no', oversca='no', trim='no', zerocor='yes', darkcor='no', flatcor='no', zero='masterbias',Stdout="/dev/null")

# <codecell>

#apall on a flat field
print " + Extract masterflat"
iraf.noao(_doprint=0,Stdout="/dev/null")
iraf.twodspec(_doprint=0,Stdout="/dev/null")
iraf.apextract(_doprint=0,Stdout="/dev/null")

# <codecell>

#adds DISPAXIS=1 to the header
iraf.hedit(images=flat, fields='DISPAXIS', value=1, add='yes', verify='no', update='yes',Stdout="/dev/null")

# <codecell>

iraf.unlearn('apall') #restores initial paramenters to apall

# <codecell>

shift_master(ap_ref[ccd], "database/ap%s" % ap_ref[ccd][:-5])

# <codecell>

check=iraf.apall(input=flat, format='multispec', referen=flat, interac='no', find='no',
                 recenter='yes', resize='no', edit='yes', trace='yes', fittrac='yes',
                 extract='yes', extras='no', review='yes', line=2000, lower=-2, upper=2, 
                 nfind=392, maxsep=45, minsep=3, width=5, radius=2, ylevel=0.3, shift='yes', 
                 t_order=7, t_niter=10, t_low_r=3, t_high_r=3, t_sampl='1:4095', t_nlost=1, 
                 npeaks=392, bkg='no', b_order=7, nsum=-10,Stdout=1)

# <codecell>

flat

# <codecell>

#create flats with different aperture widths
iraf.apall(input=flat, output='flat_1', referen=flat, format='multispec', interac='no', find='no', recenter='no', resize='yes', edit='no', trace='no', fittrac='no', extract='yes', extras='no', review='no', line=2000, lower=-1, upper=1, bkg='no', nsum=-10, ylevel="INDEF", llimit=-1, ulimit=1,Stdout="/dev/null")
# iraf.apall(input='tmp/masterflat.fits', output='tmp/flat_2', referen='tmp/masterflat.fits', format='multispec', interac='no', find='no', recenter='no', resize='yes', edit='no', trace='no', fittrac='no', extract='yes', extras='no', review='no', line=2000, lower=-2, upper=2, bkg='no', nsum=-10, ylevel="INDEF", llimit=-2, ulimit=2,Stdout="/dev/null")
# iraf.apall(input='tmp/masterflat.fits', output='tmp/flat_3', referen='tmp/masterflat.fits', format='multispec', interac='no', find='no', recenter='no', resize='yes', edit='no', trace='no', fittrac='no', extract='yes', extras='no', review='no', line=2000, lower=-3, upper=3, bkg='no', nsum=-10, ylevel="INDEF", llimit=-3, ulimit=3,Stdout="/dev/null")

# <codecell>

#normalize different flats
iraf.noao(_doprint=0,Stdout="/dev/null")
iraf.onedspec(_doprint=0,Stdout="/dev/null")
iraf.continuum(input=flat, output='flat_1_norm', lines='*', bands='*', type='ratio', wavescale='no', interac='no', sample='1:4095', functio='spline3', order=13, low_rej=2, high_rej=2, niter=10,Stdout="/dev/null")
# iraf.continuum(input='tmp/flat_2', output='tmp/flat_2_norm', lines='*', bands='*', type='ratio', wavescale='no', interac='no', sample='1:4095', functio='spline3', order=13, low_rej=2, high_rej=2, niter=10,Stdout="/dev/null")
# iraf.continuum(input='tmp/flat_3', output='tmp/flat_3_norm', lines='*', bands='*', type='ratio', wavescale='no', interac='no', sample='1:4095', functio='spline3', order=13, low_rej=2, high_rej=2, niter=10,Stdout="/dev/null")

# <codecell>

#combine normalized flats
# iraf.images(_doprint=0,Stdout="/dev/null")
# iraf.imutil(_doprint=0,Stdout="/dev/null")
# iraf.imarith(operand1='tmp/flat_1_norm', op='*', operand2=0.43, result='tmp/flat_1_norm2',Stdout="/dev/null")
# iraf.imarith(operand1='tmp/flat_2_norm', op='*', operand2=0.41, result='tmp/flat_2_norm2',Stdout="/dev/null")
# iraf.imarith(operand1='tmp/flat_3_norm', op='*', operand2=0.16, result='tmp/flat_3_norm2',Stdout="/dev/null")

# <codecell>

iraf.imsum(input='*norm.fits', output='masterflat_norm.fits', option='sum',Stdout="/dev/null")

# <headingcell level=3>

# arcs

# <codecell>

arc = '20aug10052.fits'
obj = '20aug10053.fits'

# <codecell>

#find arcs
#     print " + Reducing arcs"
#     cur.execute("SELECT spec_path,name from fields where ymd=%s and ccd='%s' and obstype='ARC' and plate='%s'" % (ymd, ccd, plate))
#     arcs=cur.fetchall()

#     #find object frames
#     cur.execute("SELECT spec_path,name from fields where ymd=%s and ccd='%s' and obstype='OBJECT' and plate='%s'" % (ymd, ccd, plate))
#     objs=cur.fetchall()

#     #make a new list of objects if a list of exposures is given:
#     objs_new=[]
#     for i in objs:
#         #print i[0][-9:-5]
#         if (str(i[0][-9:-5]) in list_of_exp) or (list_of_exp==['*']): objs_new.append(i)
#     else: pass

#     #print objs_new, list_of_exp
#     objs=objs_new

#     #copy arcs to tmp folder
#     for i in arcs:
#         system("cp %s tmp/arcs/" % (i[0]))

#     #copy objs to tmp folder
#     for i in objs:
#         system("cp %s tmp/objs/" % (i[0]))

# <codecell>

#reduce arcs
iraf.noao(_doprint=0,Stdout="/dev/null")
iraf.imred(_doprint=0,Stdout="/dev/null")
iraf.ccdred(_doprint=0,Stdout="/dev/null")

# <codecell>

iraf.ccdproc(images=arc, ccdtype='', fixpix='no', oversca='no', trim='no', zerocor='yes', darkcor='no', flatcor='no', zero='masterbias',Stdout="/dev/null")

# <codecell>

#reduce object frames
print " + Reducing objects"
iraf.hedit(images=obj, fields='DISPAXIS', value=1, add='yes', verify='no', update='yes',Stdout="/dev/null")
iraf.ccdproc(images=obj, ccdtype='', fixpix='no', oversca='no', trim='no', zerocor='yes', darkcor='no', flatcor='no', zero='masterbias',Stdout="/dev/null")

# <codecell>

#fit and subtract scattered light
print " + Fitting and removing scattered light"
iraf.apscat1.functio='cheb'
iraf.apscat1.sample='10:4090'
iraf.apscat1.order=9
iraf.apscat1.low_rej=4
iraf.apscat1.high_rej=1.1
iraf.apscat1.niter=10
iraf.apscat1.grow=0
iraf.apscat1.naverage=-1

iraf.apscat2.functio='cheb'
iraf.apscat2.order=5
iraf.apscat2.sample='1:4095'
iraf.apscat2.low_rej=3
iraf.apscat2.high_rej=3
iraf.apscat2.niter=5
iraf.apscat2.grow=1
iraf.apscat2.naverage=-70

# <codecell>

iraf.imcopy(input=obj, output='sct',Stdout="/dev/null")
iraf.apscatter(input=obj, output='', referen='masterflat_norm', interac='no', find='no', recente='no', resize='no', edit='no', trace='no', fittrace='no', subtrac='yes', smooth='yes', fitscat='yes', fitsmoo='yes', buffer=0.7)

# <codecell>

extract(arc, obj, ccd)

# <codecell>

ccd

# <headingcell level=3>

# IRAF CC tests

# <codecell>

import pyfits as pf
import pylab as plt

# <codecell>

cd ~/Documents/HERMES/reductions/iraf/HD1581/cam1/

# <codecell>

cd ~/Documents/workspace/GAP/IrafReduction/results/140820/norm/

# <codecell>

a = pf.open('20aug10053.fits')

# <codecell>

cd ~/Documents/workspace/GAP/IrafReduction/results/140821/norm/

# <codecell>

b = pf.open('22aug10038.fits')

# <codecell>

import numpy as np
def get_wl(header, app):
    WS = 'WS_'+str(app)
    WD = 'WD_'+str(app)

    first_px = float(header[WS])
    disp = float(header[WD])
    length = header['NAXIS1']
    wl = np.arange(length)*disp
    wl += first_px
    
    return wl
    

# <codecell>

def plot_app(fits1, fits2 ,app):
    plt.plot(get_wl(fits1[0].header, app),fits1[0].data[app])
    plt.plot(get_wl(fits2[0].header, app),fits2[0].data[app])
    plt.show()
    
#     plt.plot(get_wl(fits1[0].header,app), '.' )
#     plt.plot(get_wl(fits2[0].header,app), '.')
#     plt.show()
    
#     wl1=get_wl(fits1[0].header,app)
#     wl2=get_wl(fits2[0].header,app)    
#     print wl1-wl2

# <codecell>

plot_app(a,b,173)
# plot_app(b,223)

# <codecell>


# <codecell>

import RVTools as RVT
from scipy import signal

# <codecell>

flux1 = a[0].data[173]
wl1 = get_wl(a[0].header, 173)
flux2b = b[0].data[173]
wl2 = get_wl(b[0].header, 173)

# flux2 = np.interp(wl2, wl2, flux2b)
flux2 = np.interp(wl1, wl2, flux2b)
flux1 = np.interp(wl1, wl1, flux1)

# <codecell>

plt.plot(wl2,flux2b,marker='+')
plt.scatter(wl1,flux2, marker = '+', s=200, c='r')
plt.show()

# <codecell>

flux2

# <codecell>

plt.plot(signal.fftconvolve(flux1, flux2[::-1], mode='same'))
plt.show()

# <codecell>

corrHWidth =3

# <codecell>


CCCurve = []
CCCurve = signal.fftconvolve(flux1[-np.isnan(flux1)], flux2[-np.isnan(flux2)][::-1], mode='same')
corrMax = np.where(CCCurve==max(CCCurve))[0][0]
p_guess = [corrMax,corrHWidth]
x_mask = np.arange(corrMax-corrHWidth, corrMax+corrHWidth+1)
if max(x_mask)<len(CCCurve):
#                 try:
#                 print '4 params',p_guess, x_mask, np.sum(x_mask), CCCurve.shape
    p = RVT.fit_gaussian(p_guess, CCCurve[x_mask], np.arange(len(CCCurve))[x_mask])[0]
    if np.modf(CCCurve.shape[0]/2.0)[0]>1e-5:
        pixelShift = (p[0]-(CCCurve.shape[0]-1)/2.) #odd number of elements
    else:
        pixelShift = (p[0]-(CCCurve.shape[0])/2.) #even number of elements
#                 except:
#                     pixelShift = 0

#     thisQ, thisdRV = QdRV(thisCam.wavelengths[epoch], thisCam.red_fluxes[epoch])

    mid_px = flux1.shape[0]/2
    dWl = (wl1[mid_px+1]-wl1[mid_px]) / wl1[mid_px]
    RV = dWl * pixelShift * RVT.constants.c 
    print 'RV',RV

# <codecell>

cd /Users/Carlos/Documents/HERMES/reductions/6.5/HD285507_1arc/npy/

# <codecell>

data = np.load('data.npy')

# <codecell>

data

# <codecell>

a1 = a[0]

# <codecell>

a1.header.items()

# <codecell>

'Giany01' in a1.header['APID*']

# <codecell>

starNames = []
for fib in a1.header['APID*']:
#     print int(fib.key[4:])-1 ,fib.value,
    if not (('PARKED' in fib.value) or ('Grid_Sky' in fib.value) or ('FIBRE ' in fib.value)): 
        if fib.value.split(' ')[0]=='Giant01':
            print fib.key[4:]

# <codecell>

starNames

# <codecell>

aaa.value

# <codecell>


