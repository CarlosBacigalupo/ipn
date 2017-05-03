# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=3>

# Creates a list of aliases from all .obj files in a folder
# (skips the red*.obj files)

# <codecell>

from astroquery.simbad import Simbad
from astropy import coordinates
import astropy.units as u
import ads
import numpy as np
import glob
import pickle

# <codecell>

cd ~/Documents/HERMES/reductions/6.5/HD1581/obj/

# <codecell>

cd ~/Documents/HERMES/reductions/6.5/m67_lr/obj/

# <headingcell level=6>

# query by region

# <codecell>

objList = glob.glob('*.obj')

r = 5 * u.arcsecond
starAlias = [] 

for thisObj in objList[:]:
    if 'red' not in thisObj:
        filehandler = open(thisObj, 'r')
        thisStar = pickle.load(filehandler)
        star = thisStar.name
        print thisStar.name,':',
        
        gc = coordinates.SkyCoord(ra=thisStar.RA*u.si.degree, dec=thisStar.Dec*u.si.degree)

        result_table = Simbad.query_region(gc, radius=r)
#         print result_table
        if len(result_table)==1:
            
            result_table_IDs = Simbad.query_objectids(result_table[0][0])
            for CRID in result_table_IDs.columns[0]:
                starAlias.append([star,CRID])
                print CRID, '|',
#             print 
#             papers = ads.SearchQuery(q=result_table[0][0], sort="citation_count", rows=5 )
#             ads.config.token = 'el4YBpYbJoxjrpfht35uS6X7f5syc5vJNxRygvmA'
#             for paper in papers:
#                 print(str(paper.title[0]))
        else:
            print len(result_table), 'results'
        filehandler.close()
        thisStar = None
        
        
        print 
        print 
        
np.save('../starAlias.npy',starAlias)

# <headingcell level=3>

# ADS and other random tests to follow

# <codecell>

papers = ads.SearchQuery(q="supernova", sort="citation_count", rows=5 )
ads.config.token = 'el4YBpYbJoxjrpfht35uS6X7f5syc5vJNxRygvmA'

# <codecell>

for paper in papers:
    print(paper.title)

# <codecell>

# customSimbad = Simbad()
# customSimbad.get_votable_fields()
Simbad.list_votable_fields()
Simbad.get_field_description ('gcrv')

# <codecell>

for objID in result_table_IDs:
    print objID[0]
    temp = Simbad.query_object(objID[0])
    print temp
    temp= Simbad.query_bibcode(temp['COO_BIBCODE'].data.data[0])
    print temp[0].as_void()[0].split('\n')[3]

# <codecell>

temp= Simbad.query_object(result_table[0][0])
print temp
temp= Simbad.query_bibcode(temp['COO_BIBCODE'].data.data[0])
print temp

# <codecell>

temp[0].as_void()[0].split('\n')[3]

# <codecell>

result_table['COO_BIBCODE']

