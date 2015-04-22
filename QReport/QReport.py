# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
QReport.py
Creates a summary of GALAH data quality for a given date. 
'''

#########################
#Globals

out_folder = ''
data_folder = ''
emails = ''
log_file = out_folder + 'report.log'

#End of globals
##########################



okGo = True

#tests imports
try:
    print 'Importing glob module'
    import glob
except:
    print 'Imports failed. Module missing.'
    okGo=False
    

#tests report modules
try:
    pass
except: 
    okGo=False


#checks for data
try:
    pass
except:
    okGo=False


#open log
try:
    

if okGo==True:
    pass
#creates reports
#    core report

#    other reports


#emails results



else:
    print 'Initial checks failed. Report not created'

# <codecell>


