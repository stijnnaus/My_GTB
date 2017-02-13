# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 11:01:53 2016

@author: naus010
"""

import sys
import os
from numpy import array
from pylab import *
#from pyhdf.SD import *
from datetime import datetime
from datetime import timedelta
from copy import *
from numpy import linalg
from scipy import optimize
import netCDF4 as nc
from netCDF4 import Dataset


fi = 'C:\\Users\\Stijn\Documents\\Wageningen\\station_file_004.nc'
f = Dataset(fi,'r')

def read_mcf_emi(fil):
    f = open(fil,'r')
    ye = []
    rapid = []
    medium = []
    slow = []
    stock = []
    for line in f.readlines():
       ye.append(int(line.split()[0]))
       rapid.append(float(line.split()[1]))
       medium.append(float(line.split()[2]))
       slow.append(float(line.split()[3]))
       stock.append(float(line.split()[4]))
    ye = array(ye)
    rapid = array(rapid)*1e6
    medium = array(medium)*1e6
    slow = array(slow)*1e6
    stock = array(stock)*1e6
    prod = rapid+medium+slow+stock
    em0 = []
    for year in range(styear,2009):
       # calculate this year's emissions:
       iyear = year-1951   # index in arrays rapid, medium, slow, stock
       # emissions independent of fe (factor for stockpiling):
       em = 0.75*rapid[iyear] + 0.25*rapid[iyear-1] \
             + 0.25*medium[iyear] + 0.75*medium[iyear-1] + \
             0.25*slow[iyear-1] + 0.75*slow[iyear-2]
       for yearb in range(year-1,year-11,-1):
          jyear = yearb - 1951
          em += 0.1*stock[jyear] 
       em0.append(em)
    return rapid,medium,slow,stock,array(em0),prod
    
def extend_mcf_emi(rapid,medium,slow,stock,em0,edyear):
    '''
    Extend the emissions up to edyear from McCullogh using a 20% decrease per 
    year after 2008 in all emission categories.
    '''
    rapidi,mediumi,slowi,stocki = rapid[-1],medium[-1],slow[-1],stock[-1]
    for yr in range(2009,edyear):
        rapidi,mediumi,slowi,stocki = .8*rapidi, .8*mediumi, .8*slowi, .8*stocki
        rapid = np.append(rapid, rapidi)
        medium = np.append(medium, mediumi)
        slow = np.append(slow, slowi)
        stock = np.append(stock, stocki)
    prod = rapid+medium+slow+stock
    for year in range(2009,edyear):
       # calculate this year's emissions:
       iyear = year-1951   # index in arrays rapid, medium, slow, stock
       # emissions independent of fe (factor for stockpiling):
       em = 0.75*rapid[iyear] + 0.25*rapid[iyear-1] \
             + 0.25*medium[iyear] + 0.75*medium[iyear-1] + \
             0.25*slow[iyear-1] + 0.75*slow[iyear-2]
       for yearb in range(year-1,year-11,-1):
          jyear = yearb - 1951
          em += 0.1*stock[jyear] 
       em0 = np.append(em0, em)
    return rapid,medium,slow,stock,em0,prod
        

def read_ch4_emi(fil):
    pass

def read_d13C_emi(fil):
    pass

def read_d13C_obs(fil):
    f = open(fil)
    d13C, d13C_e = [],[]
    for line in f.readlines():
        spl = line.split(' ')
        if spl[0][0] != '#':
            d13C.append(float(spl[1]))
            d13C_e.append(float(spl[2]))
    return array(d13C),array(d13C_e)
    
def read_mcf_measurements():
    t_mhd_gage, mcf_mhd_gage, mcf_mhd_gage_e = gage(0,'ch3ccl3')
    t_mhd_agage, mcf_mhd_agage, mcf_mhd_agage_e   = agage(0,'ch3ccl3')
    t_thd_gage, mcf_thd_gage, mcf_thd_gage_e   = gage(1,'ch3ccl3')
    t_thd_agage, mcf_thd_agage, mcf_thd_agage_e   = agage(1,'ch3ccl3')
    t_rpb_gage, mcf_rpb_gage, mcf_rpb_gage_e   = gage(2,'ch3ccl3')
    t_rpb_agage, mcf_rpb_agage, mcf_rpb_agage_e   = agage(2,'ch3ccl3')
    t_smo_gage, mcf_smo_gage, mcf_smo_gage_e   = gage(3,'ch3ccl3')
    t_smo_agage, mcf_smo_agage, mcf_smo_agage_e   = agage(3,'ch3ccl3')
    t_cgo_gage, mcf_cgo_gage, mcf_cgo_gage_e   = gage(4,'ch3ccl3')
    t_cgo_agage, mcf_cgo_agage, mcf_cgo_agage_e   = agage(4,'ch3ccl3')

    t_glob_mcf = []    # time
    mcf_glob = []  # concentration
    mcf_glob_e = []  # error
    mcf_glob_mo = []
    mcf_glob_mo_e = []
    for year in range(1988,2009):
       mcf_mo,mcf_mo_e = [],[] # monthly averages
       for month in range(1,13):
          glob_mcf = []
          glob_mcf_e = []
          not_present = []
          tx = datetime(year,month,1,0,0)
          try:
             idx = t_mhd_agage.index(tx)
             glob_mcf.append(mcf_mhd_agage[idx])
             glob_mcf_e.append(mcf_mhd_agage_e[idx])
          except:
             try:
                idx = t_mhd_gage.index(tx)
                glob_mcf.append(mcf_mhd_gage[idx])
                glob_mcf_e.append(mcf_mhd_gage_e[idx])
             except:
                not_present.append('mhd')
                None
          try:
             idx = t_thd_agage.index(tx)
             glob_mcf.append(mcf_thd_agage[idx])
             glob_mcf_e.append(mcf_thd_agage_e[idx])
          except:
             try:
                idx = t_thd_gage.index(tx)
                glob_mcf.append(mcf_thd_gage[idx])
                glob_mcf_e.append(mcf_thd_gage_e[idx])
             except:
                not_present.append('thd')
                None
          try:
             idx = t_rpb_agage.index(tx)
             glob_mcf.append(mcf_rpb_agage[idx])
             glob_mcf_e.append(mcf_rpb_agage_e[idx])
          except:
             try:
                idx = t_rpb_gage.index(tx)
                glob_mcf.append(mcf_rpb_gage[idx])
                glob_mcf_e.append(mcf_rpb_gage_e[idx])
             except:
                not_present.append('rpb')
                None
          try:
             idx = t_smo_agage.index(tx)
             glob_mcf.append(mcf_smo_agage[idx])
             glob_mcf_e.append(mcf_smo_agage_e[idx])
          except:
             try:
                idx = t_smo_gage.index(tx)
                glob_mcf.append(mcf_smo_gage[idx])
                glob_mcf_e.append(mcf_smo_gage_e[idx])
             except:
                not_present.append('smo')
                None
          try:
             idx = t_cgo_agage.index(tx)
             glob_mcf.append(mcf_cgo_agage[idx])
             glob_mcf_e.append(mcf_cgo_agage_e[idx])
          except:
             try:
                idx = t_cgo_gage.index(tx)
                glob_mcf.append(mcf_cgo_gage[idx])
                glob_mcf_e.append(mcf_cgo_gage_e[idx])
             except:
                not_present.append('cgo')
                None
          # check whether monthly averages are present: if so give thd and mhd each half of the weight
          if len(not_present) == 0:
             mcf_glob.append(0.125*glob_mcf[0] + 0.125*glob_mcf[1] + 0.25*glob_mcf[2] + \
                   0.25*glob_mcf[3] + 0.25*glob_mcf[4])
             mcf_glob_e.append(sqrt((0.125*glob_mcf_e[0])**2 + \
                                    (0.125*glob_mcf_e[1])**2 + \
                                    (0.25 *glob_mcf_e[2])**2 + \
                                    (0.25 *glob_mcf_e[3])**2 + \
                                    (0.25 *glob_mcf_e[4])**2))
             t_glob_mcf.append(tx)
 
          # we can do without either California (thd) or Mace Head (mhd)
          elif len(not_present) == 1:
             if not_present[0] == 'thd' or not_present[0] == 'mhd':
                mcf_glob.append(array(glob_mcf).mean())
                t_glob_mcf.append(tx)
                mcf_glob_e.append(sqrt((0.25 *glob_mcf_e[0])**2 + \
                                       (0.25 *glob_mcf_e[1])**2 + \
                                       (0.25 *glob_mcf_e[2])**2 + \
                                       (0.25 *glob_mcf_e[3])**2))
          if len(not_present)<2:
              mcf_mo.append(mcf_glob[-1])
              mcf_mo_e.append(mcf_glob_e[-1])
          else:
              mcf_mo.append('mis')
              mcf_mo_e.append('mis')
       mcf_glob_mo.append( mcf_mo )
       mcf_glob_mo_e.append( mcf_mo_e )
    
    mcf_glob_mo, mcf_glob_mo_e = fill_miss(mcf_glob_mo, mcf_glob_mo_e)
    mcf_glob_yr, mcf_glob_yr_e = year_av(mcf_glob_mo, mcf_glob_mo_e)
    return mcf_glob_yr, mcf_glob_yr_e


def read_ch4_measurements():
    t_mhd_gage, ch4_mhd_gage, ch4_mhd_gage_e = gage(0,'ch4')
    t_mhd_agage, ch4_mhd_agage, ch4_mhd_agage_e   = agage(0,'ch4')
    t_thd_gage, ch4_thd_gage, ch4_thd_gage_e   = gage(1,'ch4')
    t_thd_agage, ch4_thd_agage, ch4_thd_agage_e   = agage(1,'ch4')
    t_rpb_gage, ch4_rpb_gage, ch4_rpb_gage_e   = gage(2,'ch4')
    t_rpb_agage, ch4_rpb_agage, ch4_rpb_agage_e   = agage(2,'ch4')
    t_smo_gage, ch4_smo_gage, ch4_smo_gage_e   = gage(3,'ch4')
    t_smo_agage, ch4_smo_agage, ch4_smo_agage_e   = agage(3,'ch4')
    t_cgo_gage, ch4_cgo_gage, ch4_cgo_gage_e   = gage(4,'ch4')
    t_cgo_agage, ch4_cgo_agage, ch4_cgo_agage_e   = agage(4,'ch4')

    t_glob_ch4 = []    # time
    ch4_glob = []  # concentration
    ch4_glob_e = []  # error
    ch4_glob_mo, ch4_glob_mo_e = [], []
    for year in range(1988,2009):
       ch4_mo, ch4_mo_e = [], []
       for month in range(1,13):
          glob_ch4 = []
          glob_ch4_e = []
          not_present = []
          tx = datetime(year,month,1,0,0)
          try:
             idx = t_mhd_agage.index(tx)
             glob_ch4.append(ch4_mhd_agage[idx])
             glob_ch4_e.append(ch4_mhd_agage_e[idx])
          except:
             try:
                idx = t_mhd_gage.index(tx)
                glob_ch4.append(ch4_mhd_gage[idx])
                glob_ch4_e.append(ch4_mhd_gage_e[idx])
             except:
                not_present.append('mhd')
                None
          try:
             idx = t_thd_agage.index(tx)
             glob_ch4.append(ch4_thd_agage[idx])
             glob_ch4_e.append(ch4_thd_agage_e[idx])
          except:
             try:
                idx = t_thd_gage.index(tx)
                glob_ch4.append(ch4_thd_gage[idx])
                glob_ch4_e.append(ch4_thd_gage_e[idx])
             except:
                not_present.append('thd')
                None
          try:
             idx = t_rpb_agage.index(tx)
             glob_ch4.append(ch4_rpb_agage[idx])
             glob_ch4_e.append(ch4_rpb_agage_e[idx])
          except:
             try:
                idx = t_rpb_gage.index(tx)
                glob_ch4.append(ch4_rpb_gage[idx])
                glob_ch4_e.append(ch4_rpb_gage_e[idx])
             except:
                #print 'try intepolation from smo and thc/mhd'
                try:
                   idxsmo = t_smo_gage.index(tx)
                except:
                   idxsmo = -1
                try:
                   idxmhd = t_mhd_gage.index(tx)
                except:
                   idxmhd = -1
                try:
                   idxthd = t_thd_gage.index(tx)
                except:
                   idxthd = -1
                #print idxsmo,idxmhd,idxthd
# only interpolate if both smo and one of the NH stations has valid measurement:
                if idxsmo < 0 or (idxmhd + idxthd) < 0:
                   not_present.append('rpb')
                else:
                   ch4s = ch4_smo_gage[idxsmo]
                   ch4s_e = ch4_smo_gage_e[idxsmo]
                   if idxmhd > 0 and idxthd > 0:
                      ch4a = (ch4_mhd_gage[idxmhd] + ch4_thd_gage[idxthd])*0.5
                      ch4a_e = (ch4_mhd_gage_e[idxmhd] + ch4_thd_gage_e[idxthd])*0.5
                   elif idxmhd > 0:
                      ch4a = ch4_mhd_gage[idxmhd] 
                      ch4a_e = ch4_mhd_gage_e[idxmhd]
                   else:
                      ch4a = ch4_thd_gage[idxthd] 
                      ch4a_e = ch4_thd_gage_e[idxthd]
                   glob_ch4.append(0.5*(ch4a+ch4s))
                   glob_ch4_e.append(ch4a_e + ch4s_e)   # add errors to stress interpolation
          try:
             idx = t_smo_agage.index(tx)
             glob_ch4.append(ch4_smo_agage[idx])
             glob_ch4_e.append(ch4_smo_agage_e[idx])
          except:
             try:
                idx = t_smo_gage.index(tx)
                glob_ch4.append(ch4_smo_gage[idx])
                glob_ch4_e.append(ch4_smo_gage_e[idx])
             except:
                not_present.append('smo')
                None
          try:
             idx = t_cgo_agage.index(tx)
             glob_ch4.append(ch4_cgo_agage[idx])
             glob_ch4_e.append(ch4_cgo_agage_e[idx])
          except:
             try:
                idx = t_cgo_gage.index(tx)
                glob_ch4.append(ch4_cgo_gage[idx])
                glob_ch4_e.append(ch4_cgo_gage_e[idx])
             except:
                not_present.append('cgo')
                None
          # check whether monthly averages are present: if so give thd and mhd each half of the weight
          if len(not_present) == 0:
             ch4_glob.append(0.125*glob_ch4[0] + 0.125*glob_ch4[1] + 0.25*glob_ch4[2] + \
                   0.25*glob_ch4[3] + 0.25*glob_ch4[4])
             ch4_glob_e.append(sqrt((0.125*glob_ch4_e[0])**2 + \
                                    (0.125*glob_ch4_e[1])**2 + \
                                    (0.25 *glob_ch4_e[2])**2 + \
                                    (0.25 *glob_ch4_e[3])**2 + \
                                    (0.25 *glob_ch4_e[4])**2))
             t_glob_ch4.append(tx)
 
          # we can do without either California (thd) or Mace Head (mhd)
          elif len(not_present) == 1:
             if not_present[0] == 'thd' or not_present[0] == 'mhd':
                ch4_glob.append(array(glob_ch4).mean())
                ch4_glob_e.append(sqrt((0.25 *glob_ch4_e[0])**2 + \
                                       (0.25 *glob_ch4_e[1])**2 + \
                                       (0.25 *glob_ch4_e[2])**2 + \
                                       (0.25 *glob_ch4_e[3])**2))
                t_glob_ch4.append(tx)
                
          if len(not_present) < 2:
              ch4_mo.append(ch4_glob[-1])
              ch4_mo_e.append(ch4_glob_e[-1])
          else:
              ch4_mo.append('mis')
              ch4_mo_e.append('mis')
       ch4_glob_mo.append( ch4_mo )
       ch4_glob_mo_e.append( ch4_mo_e )
    
    ch4_glob_mo, ch4_glob_mo_e = fill_miss(ch4_glob_mo, ch4_glob_mo_e)
    ch4_glob_yr, ch4_glob_yr_e =   year_av(ch4_glob_mo, ch4_glob_mo_e)
    
    return ch4_glob_yr, ch4_glob_yr_e

def fill_miss(data,data_e):
    '''
    Fills in the missing value in a month by linearly interpolating between
    the same month in the next and previous closest year that do have data.
    data: nyear x nmonth array/list of concentrations/emissions
    data_e: nyear x nmonth array/list of errors in data
    '''
    nyear = len(data)
    for y in range(nyear):
        for m in range(12):
            if data[y][m] == 'mis':
                prev, nex =  'mis','mis'
                ipr,ine = 0,0
                while type(prev) == str and (y-ipr-1) >= 0:
                    ipr+=1
                    prev = data[y-ipr][m]
                while type(nex)  == str and (y+ine+1) < nyear:
                    ine+=1
                    nex = data[y+ine][m]
                    
                if type(nex) == str:
                    data[y][m] = prev
                    data_e[y][m] = data_e[y-ipr][m]
                elif type(prev) == str:
                    data[y][m] = nex
                    data_e[y][m] = data_e[y+ine][m]
                else:
                    data[y][m] = interp( [y], [y-ipr,y+ine], [prev,nex])
                    data_e[y][m] = sqrt(data_e[y+ine][m]**2 + data_e[y-ipr][m]**2 )
    return data, data_e
    
def year_av(data,data_e):
    '''
    Calculates the yearly averages and errors of an nyear x nmonth array.
    The error is naively chosen as the error resulting from 
    the uncertainty in each month.
    '''
    
    nyear = len(data)
    yr_av,yr_er = [],[]
    for y in range(nyear): 
        yr_av.append( mean(data[y]) )
        yr_er.append( sqrt( sum( array(data_e[y])**2 ) ) / 12. )
    return array(yr_av), array(yr_er)
    
def gage(station,spec):
   from datetime import datetime
   stations = ['mhd653n00.agage.as.cn.%s.gage.mo.dat'%(spec),\
               'cmo445n00.agage.as.cn.%s.gage.mo.dat'%(spec),\
               'rpb413n00.agage.as.cn.%s.gage.mo.dat'%(spec),\
               'smo514s00.agage.as.cn.%s.gage.mo.dat'%(spec),\
               'cgo540s00.agage.as.cn.%s.gage.mo.dat'%(spec)]
   #print 'Getting %s from GAGE station '%(spec) +stations[station]
   date = []
   trac = []
   trace = []
   if stations[station] != ' ':
      try:
         f = open(os.path.join('OBSERVATIONS','monthly',stations[station]), mode = 'r')
      except IOError,msg:
         print 'An error has occurred trying to access a file'
         print msg
         #sys.exit(2)
      lines = f.readlines()
      for i in range(len(lines)):
         line = lines.pop(0)
         if line[0:1] != 'C':
            yy = int(line[0:4])
            mm = int(line[5:7])
            day = int(line[8:10])
            hour = int(line[11:13])
            minutes = int(line[14:16])
            xtrac = float(line.split()[4])
            etrac = float(line.split()[6])
	    flag = line.split()[7]
            if float(xtrac) > 1 and flag != 'P':
               date.append(datetime(yy,mm,day,hour,minutes))
               trac.append(xtrac)
               trace.append(etrac)
      f.close()
   return date, array(trac), array(trace)

def agage(station,spec):
   from datetime import datetime
   stations = ['mhd653n00.agage.as.cn.%s.md.mo.dat'%(spec),\
               'thd441n00.agage.as.cn.%s.md.mo.dat'%(spec),\
               'rpb413n00.agage.as.cn.%s.md.mo.dat'%(spec),\
               'smo514s00.agage.as.cn.%s.md.mo.dat'%(spec),\
               'cgo540s00.agage.as.cn.%s.md.mo.dat'%(spec)]
   #print 'Getting %s from AGAGE station '%(spec)+stations[station]
   date = []
   trac = []
   trace = []
   if stations[station] != ' ':
      try:
         f = open(os.path.join('OBSERVATIONS','monthly',stations[station]), mode = 'r')
      except IOError,msg:
         print 'An error has occurred trying to access a file'
         print msg
         sys.exit(2)
      lines = f.readlines()
      for i in range(len(lines)):
         line = lines.pop(0)
         if line[0:1] != 'C':
            yy = int(line[0:4])
            mm = int(line[5:7])
            day = int(line[8:10])
            hour = int(line[11:13])
            minutes = int(line[14:16])
            xtrac = float(line.split()[4])
            etrac = float(line.split()[6])
	    flag = line.split()[7]
            if float(xtrac) > 1 and flag != 'P':
               date.append(datetime(yy,mm,day,hour,minutes))
               trac.append(xtrac)
               trace.append(etrac)
      f.close()
   return date, array(trac), array(trace)
    
def read_glob_mean(fil,sty,edy,errors=False):
    ''' 
    Reads the global mean yearly data files I produced.
    errors: True if errors are included in the file
    '''
    f=open(fil)
    yrs,vals,vals_e = [],[],[]
    j=0
    for i,line in enumerate(f.readlines()):
        if line[0] == '#': continue
        lin = line.split()
        yr = float(lin[0])
        if yr>=sty and yr<edy:
            j+=1
            yrs.append(yr)
            vals.append(float(lin[1]))
            if errors == True:
                vals_e.append(float(lin[2]))
    if errors == True: return array(yrs),array(vals),array(vals_e)
    return array(yrs),array(vals)
    
def write_results(filename,header,years,mcf,ch4,d13c,fsl,fst,fme,fch4,ed13c):
    '''
    Writes the optimized results to a separate file for future reference.
    '''
    fileloc = os.path.join(os.getcwd(), 'Data output',filename)
    f = open(fileloc, 'w')
    f.write(header+'\n')
    for i,yr in enumerate(years):
        f.write('%i\t%.3f\t%.3f\t%.3f\t'%(int(yr),mcf[i],ch4[i],d13c[i]))
        f.write('%.3f\t%.3f\t%.3f\t'%(fme[i],fsl[i],fst[i]))
        f.write('%.3f\t%.3f\n'%(fch4[i],ed13c[i]))
    f.close()
    
def write_settings(filename,header,b,clen_oh,clen_em,\
                    years,mcf_e,ch4_e,d13c_e):
    '''
    Writes the settings (prior and obs errors) of the run to a separate file
    for future reference.
    '''
    fileloc = os.path.join(os.getcwd(), 'Data output',filename)
    f = open(fileloc, 'w')
    f.write(header+'\n')
    prior_e = np.diag(b)
    mcfi_e,ch4i_e,d13ci_e,foh_e,fst_e,fsl_e,fme_e,fch4_e,ed13c_e = unpack(prior_e)
    foh_e,fst_e,fsl_e,fch4_e,ed13c_e = foh_e[0],fst_e[0],fsl_e[0],fch4_e[0],ed13c_e[0]
    f.write('# Prior settings:\n')
    f.write('# Initial errors:\n\
            MCF: %.3f ppt; CH4 %.3f ppb; d13C %.3f permil\n'%(mcfi_e,ch4i_e,d13ci_e))
    f.write('# The other errors:\n\
            foh: %.3f; fst: %.3f; fsl: %.3f; fch4: %.3f; ed13c_e: %.3f permil\n'%(foh_e,fst_e,fsl_e,fch4_e,ed13c_e))
    f.write('# Correlation lengths:\n\
            In OH: %.2f yr; in CH4 emissions: %.2f yr\n'%(clen_oh,clen_em))
    f.write('# Observation errors:\n')
    f.write('# Year\tMCF(ppt)\tCH4(ppb)\t(d13C(permil)\n')
    for i,yr in enumerate(years):
        f.write('%i\t%.3f\t%.3f\t%.3f'%(int(yr),mcf_e[i],ch4_e[i],d13c_e[i]))
        f.write('\n')
    f.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    