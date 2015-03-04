
home_dir = '/Users/Carlos/Documents/HERMES/reductions/47Tuc_core_6.2' #my laptop


####################End of User Stuff########################################
import os
import create_obj as cr_obj
import pickle
import glob
import TableBrowser as tb

os.chdir(home_dir)

#load all star names from 1st file
fileList = glob.glob('cam1/*.fits')
a = tb.FibreTable(fileList[0])
starNames=a.target[a.type=='P']

print 'Collecting data from ',len(starNames),'stars'

for i,star_name in enumerate(starNames):
    print i,star_name
    thisStar = cr_obj.star(star_name)
    
    thisStar.exposures = cr_obj.exposures()
    thisStar.exposures.load_exposures(thisStar.name)
    thisStar.exposures.calculate_baryVels(thisStar)
    thisStar.name = star_name
    file_pi = open(star_name+'.obj', 'w') 
    pickle.dump(thisStar, file_pi) 
    file_pi.close()