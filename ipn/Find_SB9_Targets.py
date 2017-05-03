#!/opt/local/bin/python
#!/opt/local/bin/python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import os
import pandas as pd
import pylab as plt

# <codecell>

os.chdir('/Users/Carlos/Documents/databases/SB9')

# <codecell>

# Description of SB9

# In all cases, "|" is the field separator.

# Main.dta
#   Field #         Description
#      1            System Number (SB8: <=1469)
#      2            1900.0 coordinates (for backward compatibility with SB8)
#      3            2000.0 coordinates
#      4            Component
#      5            Magnitude of component 1
#      6            Filter component 1
#      7            Magnitude of component 2
#      8            Filter component 2
#      9            Spectral type component 1
#     10            Spectral type component 2


# Orbits.dta
#   Field #         Description
#      1            System number
#      2            Orbit number for that system
#      3            Period (d)
#      4            error on P (d)
#      5            Periastron time (JD-2400000)
#      6            error on Periastron time
#      7            Flag on periastron time
#      8            eccentricity
#      9            error on eccentricity
#     10            argument of periastron (deg)
#     11            error on omega
#     12            K1 (km/s)
#     13            error on K1 (km/s)    
#     14            K2 (km/s)
#     15            error on K2 (km/s)
#     16            systemic velocity (km/s)
#     17            error on V0 (km/s)
#     18            rms RV1 (km/s)
#     19            rms RV2 (km/s)
#     20            #RV1
#     21            #RV2
#     22            Grade (0:poor, 5: definitive)
#     23            Bibcode
#     24            Contributor
#     25            Accessibility

# Alias.dta
#   Field #         Description
#      1            System number
#      2            Catalog name
#      3            ID in that catalog


# Reference:
# Any user of SB9 is encouraged to acknowledge the catalogue with a 
# reference to
#  "SB9: The ninth catalogue of spectroscopic binary orbits", 
#  Pourbaix D., Tokovinin A.A., Batten A.H., Fekel F.C., Hartkopf W.I., 
#  Levato H., Morrell N.I., Torres G., Udry S., 2004, 
#  Astronomy and Astrophysics, 424, 727-732.

# <codecell>

#load full database into big_df

def split_RA_Dec(RADec):
    
    thisOne = [(int(RADec[:2]),int(RADec[2:4]),float(int(RADec[4:9])/1000.),int(RADec[9:12]),int(RADec[12:14]),float(int(RADec[14:18])/100.))]
    return thisOne

c = np.genfromtxt('Main.dta', delimiter='|', usecols = 2, converters={2:split_RA_Dec})
goodC = np.zeros((len(c),6))
for row in range(len(c)):
    if type(c[row])!=float:
        for col in range(6):
            goodC[row,col] =c[row][0][col]
c=goodC


#      1            System Number (SB8: <=1469)
#      2            1900.0 coordinates (for backward compatibility with SB8)
#      3            2000.0 coordinates
#      4            Component
#      5            Magnitude of component 1
#      6            Filter component 1
#      7            Magnitude of component 2
#      8            Filter component 2
#      9            Spectral type component 1
#     10            Spectral type component 2
a=np.genfromtxt('Main.dta', delimiter='|')

main_df = pd.DataFrame({ 'No' : a[:,0],
                        'RA1' : c[:,0] ,
                        'RA2' : c[:,1] ,
                        'RA3' : c[:,2] ,
                        'Dec1' : c[:,3] ,
                        'Dec2' : c[:,4] ,
                        'Dec3' : c[:,5] ,
                        'Vmag1' : a[:,4],
                        'SpecType1' : a[:,8],
                        'Vmag1' : a[:,6],
                        'SpecType1' : a[:,9]})

# Orbits.dta
#   Field #         Description
#      1            System number
#      2            Orbit number for that system
#      3            Period (d)
#      4            error on P (d)
#      5            Periastron time (JD-2400000)
#      6            error on Periastron time
#      7            Flag on periastron time
#      8            eccentricity
#      9            error on eccentricity
#     10            argument of periastron (deg)
#     11            error on omega
#     12            K1 (km/s)
#     13            error on K1 (km/s)    
#     14            K2 (km/s)
#     15            error on K2 (km/s)
#     16            systemic velocity (km/s)
#     17            error on V0 (km/s)
#     18            rms RV1 (km/s)
#     19            rms RV2 (km/s)
#     20            #RV1
#     21            #RV2
#     22            Grade (0:poor, 5: definitive)
#     23            Bibcode
#     24            Contributor
#     25            Accessibility
b=np.genfromtxt('Orbits.dta', delimiter='|')
orbits_df = pd.DataFrame({ 'No' : b[:,0],
                        'period(days)' : b[:,2] ,
                        'peri_time' : b[:,4] ,
                        'peri_arg' : b[:,9] ,
                        'eccentricity' : b[:,7],
                        'K1' : b[:,11],
                        'K2' : b[:,13],
                        'K1_P' : b[:,11]/b[:,2],
                        'grade' : b[:,21]})


#   Field #         Description
#      1            System number
#      2            Catalog name
#      3            ID in that catalog
d0=np.genfromtxt('Alias.dta', delimiter='|',usecols = 0, converters={0:int})
d1=np.genfromtxt('Alias.dta', delimiter='|',usecols = 1, converters={1:str})
d2=np.genfromtxt('Alias.dta', delimiter='|',usecols = 2, converters={2:str})
alias_df = pd.DataFrame({ 'No' : d0,
                         'cat' : d1 ,
                         'ID' : d2})




big_df = main_df.merge(orbits_df, on='No')
big_df['SpecType1'] = big_df['SpecType1'].astype(str)

# <codecell>

big_df

# <codecell>

dailyRV_mask = ((big_df['K1_P'] > 0.5) & (big_df['K1_P'] < 10.))
period_mask = big_df['period(days)'] < 5. 
K2_mask = np.isnan(big_df['K2'])
eccentricity_mask = big_df['eccentricity'] < 0.1
grade_mask = big_df['grade'] == 5
dec_mask = ((big_df['Dec1'] > -75) & (big_df['Dec1'] < 15)) 
RA_mask = ((big_df['RA1'] > 22) | (big_df['RA1'] <4)) 
no667_mask = big_df['No'] != 667

big_df['Vmag1'][big_df['No']==40]=5.393
big_df['Vmag1'][big_df['No']==1482]=np.nan
big_df['Vmag1'][big_df['No']==1847]=np.nan

big_df['SpecType1'][big_df['No']==40]='F6V'
big_df['SpecType1'][big_df['No']==1482]=np.nan
big_df['SpecType1'][big_df['No']==1847]=np.nan

full_mask = dailyRV_mask & period_mask & eccentricity_mask & grade_mask & dec_mask & K2_mask & RA_mask & no667_mask
print 'Candidates in selection',np.sum(full_mask)
big_df[full_mask]


# <codecell>

'''observing days:
day# start end
1 2456890.083333 2456890.291667 #20th
2 2456891.083333 2456891.291667 #21
3 2456892.083333 2456892.291667 #22
4 2456894.083333 2456894.291667 #24
5 2456895.083333 2456895.291667 #25
'''

# <codecell>

#visibility plot
for i in np.where(full_mask==True)[0]:
    RA1 = big_df['RA1'][i]
    RA2 = big_df['RA2'][i]
    RA3 = big_df['RA3'][i]
    Dec1 = big_df['Dec1'][i]
    Dec2 = big_df['Dec2'][i]
    Dec3 = big_df['Dec3'][i]
    no = big_df['No'][i]
    
    RA = (RA1+RA2/60+RA3/3600)*15
    Dec = (Dec1+Dec2/60+Dec3/3600)
    print int(no), RA, Dec
    if i ==56:
        cc = 'r'
    elif i ==1981:
        cc = 'g'
    else:
        cc = 'b'

    plt.scatter(RA, Dec, label = int(no), color = cc)
plt.xlabel('RA (deg)')
plt.ylabel('Dec (deg)')
    
plt.gca().add_patch(plt.Rectangle((0,-75),4*15,90, alpha = 0.5, label = 'Observing Window'))
plt.gca().add_patch(plt.Rectangle((22*15,-75),2*15,90, alpha = 0.5))
plt.axis((0,360,-90,90))
plt.grid(True , which = 'major')
plt.xticks(np.arange(0,360, 20))
plt.yticks(np.arange(-90,90, 20))
plt.legend()
plt.show()

# <codecell>

#RV vs day plot for selected SB
start_day = 2456889.500000 # The Julian date for CE  2014 August 20 00:00:00.0 UT  (10am australia)
end_day = 2456895.500000 #The Julian date for CE  2014 August 26 00:00:00.0 UT (10am australia)

days = np.linspace(start_day, end_day) 

for i in np.where(full_mask==True)[0]:
#     print i
    P = big_df['period(days)'][i]
    peri_arg = big_df['peri_arg'][i]
    peri_time = big_df['peri_time'][i] + 2400000
    K1 = big_df['K1'][i]
    no = big_df['No'][i]
    print no, K1, peri_time, P, peri_arg
    RV = K1* np.sin( (days-peri_time)/P*2*np.pi + peri_arg/360*2*np.pi )
    plt.plot(days, RV, linewidth = 1, label = int(no) )

plt.xlabel('JD')
plt.ylabel('RV (km/s)')
plt.gca().add_patch(plt.Rectangle((2456890.083333,-100),0.2083339998498559,200, alpha = 0.5, label = 'Observing Time'))
plt.gca().add_patch(plt.Rectangle((2456891.083333,-100),0.2083339998498559,200, alpha = 0.5))
plt.gca().add_patch(plt.Rectangle((2456892.083333,-100),0.2083339998498559,200, alpha = 0.5))
plt.gca().add_patch(plt.Rectangle((2456894.083333,-100),0.2083339998498559,200, alpha = 0.5))
plt.gca().add_patch(plt.Rectangle((2456895.083333,-100),0.2083339998498559,200, alpha = 0.5))

plt.legend()
plt.show()

# <codecell>


# <codecell>

P = big_df['period'][0]
peri_arg = big_df['peri_arg'][0]
peri_time = big_df['peri_time'][0]
print peri_time
days = np.linspace(peri_time,peri_time+P) + 2400000
K1 = big_df['K1'][0]
RV = K1* np.sin(days/P*2*np.pi+peri_arg/360*2*np.pi)
plt.plot(days, RV)
plt.show()

# <codecell>

np.asarray(c)

# <codecell>

a=np.genfromtxt('Main.dta', delimiter='|')#, converters= {1: str})

# <codecell>

plt.plot(range(10))
plt.xlabel('asdad')
plt.show()

# <codecell>


