

star_name = 'HD1581'
home_dir = '/Users/Carlos/Documents/HERMES/reductions/HD1581_6.2/' #my laptop


####################End of User Stuff########################################
import os
import create_obj as cr_obj

os.chdir(home_dir)

thisStar = cr_obj.star('Giant01')
thisStar.exposures = cr_obj.exposures()
thisStar.exposures.load_exposures(thisStar.name)
thisStar.exposures.calculate_baryVels(thisStar)
thisStar.name = star_name
file_pi = open(star_name+'.obj', 'w') 
pickle.dump(thisStar, file_pi) 
file_pi.close()