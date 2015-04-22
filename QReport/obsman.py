#! /usr/bin/env python
import os
from threading import Thread
import sys
import glob
from datetime import datetime
import time
import numpy as np
import ephem
import obsutil2
import icdb

# Assumptions 
# data is stored in directories named as YYMMDD, i.e., x.isdigit()and(len(x)==6)
# no other directory should satisfy this
# Only obsmanager generated files should be named with prefix gf1_ or gf0_ .
# Other files should not use this prefix.


class Obsman():
    def __init__(self):
        self.icdb=icdb.ICDB()
        self.currentdir=''
#        self.telescope_dir='/media/Expansion Drive/telescope/'
        self.telescope_dir='/data_lxy/aatobs/OptDet_data/'
#        self.sdsdir='/media/Expansion Drive/survey/'
#        self.galahdir='/media/Expansion Drive/survey/'
        self.galahdir='/data_macb/galah/survey/'
        x=datetime.now()-datetime.utcnow()
        self.utcoffset=round(x.seconds/60.0)/60.0

    def datestr_for_today(self):
        d=str(datetime.utcnow()).split()[0].replace('-','')
        return d[2:]

    def report(self,datestr=None):
        if datestr ==  None:
            datestr=self.datestr_for_today()
        
        if datestr.isdigit() and (len(datestr)==6):
            if os.path.isdir(self.galahdir+datestr+'/'):
                g=obsutil2.GalahTable(self.galahdir+datestr+'/')
                g.show()
            else:
                print 'Directory not found:',self.galahdir+datestr
        else:
            print 'Incorrect date string, should be YYMMDD'


    def check_setup(self,datestr,verify=True):
        if os.path.isdir(self.galahdir+datestr) == False:
            print 'Directory for ',datestr,' does not exist.'
            print 'Do you want to create it?'
            temp=raw_input('Type y or n:')
            temp=temp.lower()
            if (temp.lower() == 'y')or(temp.lower() == 'yes'):
                self.setup(datestr)
                    
    def verify_update(self):
        last_verification=0
        if os.path.isfile(self.icdb.archivedir+'last_verification.txt'):
            fp=open(self.icdb.archivedir+'last_verification.txt','r')
            temp=fp.read().strip()
            fp.close()
#            temp=temp.split()[0].replace('/','')
            last_verification=int(temp)
        today=int(self.datestr_for_today())
        if last_verification != today:
            print 'Last Verification=',last_verification
            print 'Today            =',today        
            datestrs=os.listdir(self.galahdir)
            for temp in datestrs:
                temp=temp.strip()
                if temp.isdigit() and (len(temp)==6):
                    temp=int(temp)
                    if (temp>=last_verification)and(temp<today):
                        self.update(datestr=str(temp))
            fp=open(self.icdb.archivedir+'last_verification.txt','w')
            fp.write(str(today))
            fp.close()

    def check_lock(self):
        fields=self.icdb.locked_fields()
        print 'Checking for locked fields'
        print 'No of locked fields:',fields.size
        if fields.size>0:
            print 'Locked fields found', fields.size
            print 'Possible Cause: fld files where generated but an update'
            print 'was not done'
 
    def check(self):
        datestrs=os.listdir(self.galahdir)
        datestrs.sort()
        j1=0
        j2=0
        print 'Checking for unupdated fields'
        print '{0:10} {1:10} {2:10}'.format('Date','unobserved','observed')
        for datestr in datestrs:
            datestr=datestr.strip()
            if datestr.isdigit()and(len(datestr)==6):
                g=obsutil2.GalahTable(self.galahdir+datestr+'/')
                for i in g.ind_observed:
                    obsid,fieldid,srcfile,ind=g.get_observed(i)
                    if self.icdb.contains_obsid(obsid)==0:
                        print '{0:10} {1:10} {2:10}'.format(datestr,obsid,fieldid)
                        j1=j1+1
                    else:
                        j2=j2+1
                print '{0:10} {1:<10} {2:<10}'.format(datestr,j1,j2)

        print 'Unupdated fields=',j1
        print 'Updated   fields=',j2
        fields=self.icdb.locked_fields()
        print 'Checking for locked fields'
        print 'No of locked fields:',fields.size
        if fields.size>0:
            print 'Locked fields found', fields.size
            print 'Possible Cause: fld files where generated but an update'
            print 'was not done'

    def setup(self,datestr=None):
        # setup directories for todays date, note the date 
        # self.currentDate=ephem.Date(datetime.utcnow())
        # currentdir=''
        # with open(checkfile,'w') as fp:
        #     fp.write(currentdir)
        if datestr ==  None:
            datestr=self.datestr_for_today()
        print 'Running setup and Creating directories:'
        if os.path.isdir(self.galahdir+datestr) == False:
            print '\t',self.galahdir+datestr
            print '\t',self.galahdir+datestr+'/data'
            print '\t',self.galahdir+datestr+'/fld'
            print '\t',self.galahdir+datestr+'/sds'
            print '\t',self.galahdir+datestr+'/comments'
            os.mkdir(self.galahdir+datestr)
            os.mkdir(self.galahdir+datestr+'/fld')
            os.mkdir(self.galahdir+datestr+'/sds')
            os.mkdir(self.galahdir+datestr+'/data')
            os.mkdir(self.galahdir+datestr+'/comments')
            fp=open(self.galahdir+datestr+'/comments/comments_'+datestr+'.txt', 'w')
            fp.write('# fields=[run,obstatus,seeing_min,seeing_max,comment] \n')
            fp.close()
        else:
            print 'Directory already exists:',self.galahdir+datestr
            

    def print_write_fields(self,date,duration=1.0,nsize=10,datestr=None):
        # filter for sun moon airmass
        print 'Using (local-utc)=',self.utcoffset,' hrs'
        if datestr ==  None:
            datestr=self.datestr_for_today()
        self.check_setup(datestr)
#        self.verify_update()
        d=ephem.Date(date)-self.utcoffset/24.0
        if d>ephem.Date(datetime.utcnow()):
            status=obsutil2.print_fields_external(d,duration,self.galahdir+datestr+'/fld/')
            if status == 0:
                fields=self.icdb.print_fields(d,duration,nsize)
                field_id=raw_input('Name the field_id you want to write (or n to exit):')
                print '-'*88
                field_id=field_id.lower()
                if (field_id == 'n') or (field_id == 'no') or (len(field_id) == 0):
                    print 'No field_id specified exiting'
                else:
                    field_id=int(field_id)
                    if field_id in fields['field_id']:
                        outfile=self.icdb.write_fld(field_id,d,self.galahdir+datestr+'/fld/')                    
                        if outfile != None:
                            self.icdb.lock_field(field_id)
                    else:
                        print 'Typo Error: Input is not from the choice given'
        else:
            print 'Date is in past. Correct the date.'



    # def write_fld(self,field_id,date,datestr=None):
    #     # get stars in a field, write the fld, and lock the field
    #     print 'Using (local-utc)=',self.utcoffset,' hrs'
    #     if datestr ==  None:
    #         datestr=self.datestr_for_today()
    #     self.check_setup(datestr)
    #     d=ephem.Date(date)-self.utcoffset/24.0
    #     outfile=self.icdb.write_fld(field_id,d,self.galahdir+datestr+'/fld/')
    #     #data=self.icdb.get_stars(field_id)
    #     if outfile != None:
    #         self.icdb.lock_field(field_id)


    def update(self,datestr=None):
        # _sync, _update_starsleft
        if datestr ==  None:
            datestr=self.datestr_for_today()
        status=self.sync(datestr=datestr)
        if status == 0:
            print 'Updating observed fields for date,',datestr
            g=obsutil2.GalahTable(self.galahdir+datestr+'/')
            for i in g.ind_observed:
                obsid,fieldid,srcfile,ind=g.get_observed(i)
                self.icdb.update_observed(obsid,fieldid,srcfile,ind)

            self.icdb.unlock_fields()
            print 'update: success'
        else:
            raise RuntimeError('update: failure as sync failed')
#            print 'update: failure'

    def sync(self,datestr=None):
        # sync data from telescope to galah dir
        if datestr ==  None:
            datestr=self.datestr_for_today()
        destdir=self.galahdir+datestr+'/data/'
        srcdir=self.telescope_dir+datestr+'/'
        srcdir=srcdir.replace(' ','\ ')
        destdir=destdir.replace(' ','\ ')

        if os.path.isdir(destdir):        
            if os.path.isdir(srcdir):        
                print 'Transferring data to galah dir.....'
                print 'rsync -avz '+srcdir+' '+destdir
                status=os.system('rsync -avz --exclude=*/drt_* '+srcdir+' '+destdir)
            else:
                status=0
        else:
            status=1
            print 'srcdir or destdir missing:',srcdir, destdir
        if status != 0:
            print 'obsman ERROR: sync was not successful'
        return status




def _usage():
        print "NAME:"
        print '\t obsman 0.0.1  (Observation manager for galah survey)'
        print "\t Copyright (c) 2014 Sanjib Sharma"
        print "CAUTION:"
        print "\t NEVER INTERRUPT OR STOP THE PROGRAM,"
        print "\t EVEN IF YOU MAKE A MISTAKE."
        print "USAGE:"
#        print "\t obsman\t -setup  [date]"
        print "\t ./obsman.py -print  date time [nsize]"
        print "\t ./obsman.py -update [datestr]"
        print "\t ./obsman.py -help"
        print "EXAMPLE:"
        print "\t ./obsman.py -print  2014/07/22 18:00:00"
        print "\t ./obsman.py -print  2014/07/22 18:00:00 10"
        print "\t ./obsman.py -update"
        print "\t ./obsman.py -update 140707"
        print "ARGUMENTS:"
        print "Optional arguments are denoted in square brackets."
        print "Square bracket are not used when typing them in reality."
        print " datestr- Is specififed as YYMMDD e.g 140722."
        print "          Default is to use todays date."
        print " nsize  - Max no of fields to print, default is 10"
        print "DESCRIPTION:"
        print "    -print  This will print fields for the given date and time." 
        print "            The user is then give an option to write the fld file. \n" 
        print "            If the directories for the day are not yet created" 
        print "            it will prompt you to create them.\n" 
        print "            Time is local time. Offset with UT is printed "
        print "            which the user should verify. Note after midnight" 
        print "            the date should change.\n"
        print "            It also verifies if there are any missing updates"
        print "            and will do them if needed.\n"
        print "    -update Must be run after all observations are done" 
        print "            for a night. It should not be run unless  all " 
        print "            observations are finished.\n"
        print "            It will update the observed fields and will also"  
        print "            sync the data from telescope to galah dir\n"
        print "            Default is to use present date. However, previous" 
        print "            dates can be passed to it.\n"
        print "    -help   print this help message"  
        print "ADDING COMMENTS"
        print "A file YYMMDD_comments.txt is created in YYMMDD/comments/ "
        print " directory. This has following columns                    "
        print "run, obstatus, seeing_min, seeing_max, comment"
        print "One should enter the data separated by comma, e.g., "
        print " 1, 1, 1.5, 1.6, null "
        print " 1, 1, 1.5, 1.6, bad weather "
        print " 1, 1, null, null, bad weather \n"
        print "obstatus- Normally it is set to 1. If something goes wrong" 
        print "           and the data should not be used then set it to 0."
        print "           Bad weather is not a valid reason to set it to 0."
        print "EXTRA COMMANDS:"
        print "There is no need for the user to run these commands as both"
        print "print and update run them as and when needed."
        print "    -setup  To explicity generate directories for today. \n" 
        print "    -sync   To explicity sync data from telescope to galah dir\n" 
#        print "\t obsman\t -sync  [date] "
#        print "\t -setup   ","make directories for today in galah dir."  
#        print "\t           "  
#        print "\t -write   ","write fld file for the given fieldid"    
#        print "\t -sync    ","copy fits files from telescope to galah dir"    
#        print "\t sync,update,setup can take date as an optional parameter"
#        print "\t print can take nsize an integer as an optional parameter"
#        print "\t this specifies the maximum number of fields to print"
        print "CONTACT:"
        print "sanjib.sharma at gmail"
        print "Emergency phone no 0434191346"


def noInterrupt(*argv):
    om=Obsman()
    datestr=None
    if len(argv) > 1:
        if len(argv) >=3:
            datestr=argv[2].strip()
            if '/' in datestr:
                datestr=datestr.replace('/','')[2:]
        if   argv[1] == '-print':
            if len(argv) < 4:
                _usage()
            else:
                if len(argv[2]) == 10:
                    if len(argv) == 4:
                        om.print_write_fields(argv[2]+' '+argv[3])
                    else:
                        om.print_write_fields(argv[2]+' '+argv[3],nsize=int(argv[4]))
                else:
                    _usage()
        # elif argv[1] == '-write':
        #     if len(argv) != 5:
        #         _usage()
        #     else:
        #         om.write_fld(int(argv[4]),argv[2]+' '+argv[3])
        elif argv[1] == '-update':
            om.update(datestr=datestr)
        elif argv[1] == '-sync':
            status=om.sync(datestr=datestr)
        elif argv[1] == '-setup':
            om.setup(datestr=datestr)
        elif argv[1] == '-check':
            om.check()
        elif argv[1] == '-checklock':
            om.check_lock()
        elif argv[1] == '-report':
            om.report(datestr=datestr)
        else:
            _usage()
    else:
        _usage()




if __name__  ==  '__main__':
    a = Thread(target=noInterrupt,args=sys.argv)
    a.start()
    a.join()
