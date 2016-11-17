# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 11:36:16 2016

@author: Stijn
"""

import sys
import os
import time
from numpy import array
from pylab import *
#from pyhdf.SD import *
from datetime import datetime
from datetime import timedelta
from copy import *
from numpy import linalg
from scipy import optimize

def calculate_J(xp):
    x = precon_to_state(xp)
    con = forward_all(x)
    dep_mcf,J_mcf,J_list_mcf = cost_mcf(con)
    dep_c12,J_c12,J_list_c12 = cost_c12(con)
    dep_c13,J_c13,J_list_c13 = cost_c13(con)
    
    J_pri = sum(dot(b_inv, ( x - x_prior )**2)) # prior
    J_obs = J_mcf + J_c12 + J_c13 # mismatch with obs
    J_tot = .5 * ( J_pri + J_obs )
    print 'Cost function value:',J_tot
    return J_tot
    
def calculate_dJdx(xp):
    x = precon_to_state(xp)
    foh = x[:nt]
    mcf_save,c12_save,c13_save = forward_all(x)
    
    con = forward_all(x)
    dep_mcf,J_mcf,J_list_mcf = cost_mcf(con)
    dep_c12,J_c12,J_list_c12 = cost_c12(con)
    dep_c13,J_c13,J_list_c13 = cost_c13(con)
    dep = [dep_mcf,dep_c12,dep_c13]
    
    dfoh_mcf,dmcfi,dfmcf = adjoint_model_mcf( dep_mcf, foh, mcf_save )
    dfoh_c12,dc12i,df12 = adjoint_model_c12( dep_c12, foh, c12_save )
    dfoh_c13,dc13i,df13 = adjoint_model_c13( dep_c13, foh, c13_save )
    dfoh = dfoh_mcf + dfoh_c12 + dfoh_c13
    
    dJdx_obs = np.concatenate((dfoh, dmcfi, dfmcf, \
                                dc12i, df12, dc13i, df13))
    dJdx_pri = dot(b_inv, (x-x_prior))
    dJdx = dJdx_obs + dJdx_pri
    dJdxp = dot( L_adj, dJdx )
    print 'Cost function deriv:',max(dJdxp)
    return dJdxp

def adjoint_model_mcf( dep, foh, mcf_save ):
    pulse_mcf = adj_obs_oper( dep )
    dmcf,dfmcf,dfoh = zeros(nt),zeros(nt),zeros(nt)
    dmcfi = 0.
    rapidc = rapid/conv_mcf
    
    for iyear in range(edyear-1,styear-1,-1):
        i  = iyear-styear
        ie = iyear-1951
        
        # Add adjoint pulses
        dmcfi  += pulse_mcf[i]
        
        # Chemistry
        dfoh[i] = - l_mcf_oh * mcf_save[i] * dmcfi
        dmcfi   = dmcfi *  (1. - foh[i] * l_mcf_oh - l_mcf_strat - l_mcf_ocean)
        dmcf[i] = dmcfi
        
        # Emissions
        adjem = - 0.75 * rapidc[ie] * dmcf[i]
        if (iyear + 1) < edyear: adjem -= 0.25 * rapidc[ie] * dmcf[i+1]

        for yearb in range(iyear+1 , iyear+11):
            i2  = yearb - styear
            ie2 = yearb - 1951
            if yearb < edyear: adjem += 0.1 * rapidc[ie] * dmcf[i2]
        dfmcf[i] = adjem
            
    adj_mcf = [dfoh, array([dmcfi]), dfmcf]
    return adj_mcf
    
def adjoint_model_c12( dep, foh, c12_save ):
    pulse_c12 = adj_obs_oper( dep )
    em0_12 = em0_c12 / conv_ch4
    
    df12,dfoh = zeros(nt),zeros(nt)
    dc12i = 0
    
    for iyear in range(edyear-1, styear-1, -1):
        i = iyear-styear
        dc12i += pulse_c12[i]
        
        dfoh[i] = - l_ch4_oh * c12_save[i] * dc12i
        dc12i = dc12i * (1. - foh[i] * l_ch4_oh - l_ch4_other)
        
        df12[i] = em0_12[i] * dc12i
        
    adj_c12 = [dfoh, array([dc12i]), df12]
    return adj_c12
    
def adjoint_model_c13( dep, foh, c13_save ):
    pulse_c13 = adj_obs_oper( dep )
    em0_13 = em0_c13 / conv_c13
    
    df13,dfoh = zeros(nt),zeros(nt)
    dc13i = 0
    
    for iyear in range(edyear-1, styear-1, -1):
        i = iyear - styear
        dc13i += pulse_c13[i]
        
        dfoh[i] = - l_ch4_oh * c13_save[i] * dc13i * a_ch4_oh
        dc13i   = dc13i * (1 - foh[i] * l_ch4_oh * a_ch4_oh - a_ch4_other * l_ch4_other)
        
        df13[i] = em0_13[i] * dc13i
        
    adj_c13 = [dfoh, array([dc13i]), df13]
    return adj_c13

def cost_mcf(con):
    mcf,ch4,d13c = obs_oper(con)
    dif = (mcf - mcf_obs)
    dep = dif / mcf_obs_e**2
    cost_list = dif*dep
    cost = sum(cost_list)
    return dep,cost,cost_list
    
def cost_ch4(con):
    mcf,ch4,d13c = obs_oper(con)
    dif = (ch4 - ch4_obs)
    dep = dif / ch4_obs_e**2
    cost_list = dif*dep
    cost = sum(cost_list)
    return dep,cost,cost_list
    
def cost_d13c(con):
    mcf,ch4,d13c = obs_oper(con)
    dif = (d13c - d13c_obs)
    dep = dif / d13c_obs_e**2
    cost_list = dif*dep
    cost = sum(cost_list)
    return dep,cost,cost_list
    
def cost_c12(con):
    mcf,c12,c13 = obs_oper(con)
    dif = (c12 - c12_obs)
    dep = dif / c12_obs_e**2
    cost_list = dif*dep
    cost = sum(cost_list)
    return dep,cost,cost_list
    
def cost_c13(con):
    mcf,c12,c13 = obs_oper(con)
    dif = (c13 - c13_obs)
    dep = dif / c13_obs_e**2
    cost_list = dif*dep
    cost = sum(cost_list)
    return dep,cost,cost_list

def calc_mismatch(x):
    C = forward_all(x)
    C_obs = obs_oper(C)
    return C,array(C_obs - con_data)

def forward_tl_mcf(x, foh, mcf_save):
    dfoh, dmcf0, dfmcf = x[:nt], x[nt], x[nt+1:2*nt+1]
    dem = em0_mcf + mcf_shift(dfmcf)
    dem /= conv_mcf
    
    dmcfs = zeros(nt); dmcfi = dmcf0
    for year in range(styear,edyear):
        i = year - styear
        dmcfi += dem[i]
        dmcfi = dmcfi * ( 1 - l_mcf_ocean - l_mcf_strat - l_mcf_oh * foh[i] ) - \
                dfoh[i] * mcf_save[i] * l_mcf_oh
        dmcfs[i] = dmcfi
        
    return dmcfs
    
def forward_tl_c12(x, foh, c12_save):
    dfoh, dc12_0, dfc12 = x[:nt], x[2*nt+1], x[2*nt+2 : 3*nt+2]
    dem_12 = dfc12 * (em0_c12 / conv_ch4)
    
    dc12s = []; dc12 = dc12_0
    for year in range(styear,edyear):
        i = year - styear
        dc12 += dem_12[i]
        dc12  = dc12 * ( 1 - l_ch4_other - l_ch4_oh * foh[i]) - \
                dfoh[i] * c12_save[i] * l_ch4_oh
        dc12s.append(dc12)
    
    return array(dc12s)
    
def forward_tl_c13(x, foh, c13_save):
    dfoh, dc13_0, dfc13 = x[:nt], x[3*nt+2], x[3*nt+3 : 4*nt+3]
    dem_c13 = dfc13 * (em0_c13 / conv_c13)
    
    dc13s = []; dc13 = dc13_0
    for year in range(styear,edyear):
        i = year - styear
        dc13 += dem_c13[i]
        dc13  = dc13 * ( 1 - a_ch4_other * l_ch4_other - a_ch4_oh * l_ch4_oh * foh[i] ) - \
                dfoh[i] * c13_save[i] * l_ch4_oh
        dc13s.append(dc13)
    
    return array(dc13s)

def forward_all(x):
    C_mcf,C_c12,C_c13 = forward_mcf(x),forward_c12(x),forward_c13(x)
    return C_mcf,C_c12,C_c13

def forward_mcf(x):
    foh, mcf0, fmcf = x[:nt], x[nt], x[nt+1:2*nt+1]
    em = em0_mcf + mcf_shift(fmcf)
    em /= conv_mcf
    
    mcfs = []; mcf = mcf0
    for year in range(styear,edyear):
        i = year - styear
        mcf += em[i]
        mcf = mcf * ( 1 - l_mcf_ocean - l_mcf_strat - l_mcf_oh * foh[i] )
        mcfs.append(mcf)
        
    return array(mcfs)
    
def forward_c12(x):
    foh, c12_0, fc12 = x[:nt], x[2*nt+1], x[2*nt+2 : 3*nt+2]
    em_c12 = fc12 * (em0_c12 / conv_ch4)
    
    c12s = []; c12 = c12_0
    for year in range(styear,edyear):
        i = year - styear
        c12 += em_c12[i]
        c12  = c12 * ( 1 - l_ch4_other - l_ch4_oh * foh[i])
        c12s.append(c12)
    
    return array(c12s)
   
def forward_c13(x):
    foh, c13_0, fc13 = x[:nt], x[3*nt+2], x[3*nt+3 : 4*nt+3]
    em_c13 = fc13 * (em0_c13 / conv_c13)
    
    c13s = []; c13 = c13_0
    for year in range(styear,edyear):
        i = year - styear
        c13 += em_c13[i]
        c13  = c13 * ( 1 - a_ch4_other * l_ch4_other - a_ch4_oh * l_ch4_oh * foh[i] )
        c13s.append(c13)
    
    return array(c13s)

def mcf_shift(f_mcf):
    '''
    Computes the shift in emissions that results from repartitioning of the production
    f_mcf: the fraction of rapid emissions that is instead stockpiled (ntx1 array)
    ''' 
    shifts = []
    for year in range(styear,edyear):
        fyear = year - styear # Index in f_mcf
        eyear = year - 1951 # Index in emissions
        shift = - 0.75 * rapid[eyear] * f_mcf[fyear]
        if (fyear - 1) >= 0: shift -= 0.25 * rapid[eyear-1] * f_mcf[fyear-1]
        # Stockpiling
        for yearb in range(year-1,year-11,-1):
            fyear2 = yearb - styear  
            eyear2 = yearb - 1951
            if fyear2 >= 0: shift += 0.1*f_mcf[fyear2]*rapid[eyear2]
        shifts.append(shift)
    
    return array(shifts)
    
def obs_oper(con):
#    mcf = con[0]
#    c12 = con[1]
#    c13 = con[2]
#    ch4,d13c = split_to_deltot(c12,c13)
#    obs = array([mcf,ch4,d13c])
#    return obs
    return con

def adj_obs_oper(dep):
#    mcf = dep[0]
#    ch4 = dep[1]
#    d13c = dep[2]
#    c12,c13 = deltot_to_split(ch4,d13c)
#    adj = array([mcf,c12,c13])
#    return adj
    return dep
    
def state_to_precon(x):
    '''
    Convert the state and derivative to the preconditioned space.
    '''
    return dot( L_inv, (x - x_prior) ) 

def precon_to_state(xp):
    '''
    Convert the preconditioned state to the original space.
    '''
    return dot( L_precon, xp ) + x_prior

def split_to_deltot(c12, c13):
    '''
    Converts split c12 and c13 values (emission/concentration)  
    to a delta 13C value and a total CH4 quantity.
    '''
    R_sample = c13 / c12
    delCH4 = ( R_sample / R_ST ) - 1
    totCH4 = c12 + c13
    return totCH4, 1000*delCH4

def deltot_to_split(totCH4, delc, totCH4_e = [None], delc_e = [None],mass=False):
    '''
    Converts a (list of) d13C and total CH4 values to c12 and c13.
    If abso = True, then the total CH4 input is in mass (eg emissions), and a 
    correction is needed, since 1 mol c12 has a different mass than 1 mol
    c13.
    '''
    if type(delc) == float:
        q = R_ST * ( delc/1000. + 1 )
        c12 = totCH4 / (1+q)
        c13 = q * c12
        return c12, c13
        
    totCH4, delc = array(totCH4), array(delc)/1000.
    q = array( R_ST * (delc+1) )
    c12 = totCH4 / (q+1)
    c13 = q*c12
    if mass:
        c13 *= (conv_c13 / conv_ch4)
    
    if delc_e[0] == None or totCH4_e[0] == None:
        return c12, c13
    else: 
        totCH4_e, delc_e = array(totCH4_e), array(delc_e)/1000.
        c12_e = sqrt((   totCH4_e / (1+q) )**2 + \
                    ( delc_e * totCH4 * R_ST / (1+q)**2 )**2)
        c13_e = sqrt(( q*totCH4_e / (1+q) )**2 + \
                    ( delc_e * totCH4 * R_ST * (1/(1+q) - q/(1+q)**2) )**2)
        if mass:
            c13_e *= (conv_c13 / conv_ch4)
        return c12, c13, c12_e, c13_e
    

# setup model variables, emissions, etc:
m = 5e18
xmair = 28.5
xmcf = 133.5
xch4 = 16.0
xc13 = 17.0
conv_mcf = xmcf / 10**12 * m / xmair # kg/ppt
conv_ch4 = xch4 / 10**9  * m / xmair # kg/ppb
conv_c13 = xc13 / 10**9  * m / xmair # kg/ppb
l_mcf_ocean = 1./83.0  # loss rate yr-1
l_mcf_strat = 1./45.0
oh = 0.70e6  # molecules/cm3
temp = 272.0  # Kelvin        #
l_mcf_oh = (1.64e-12*exp(-1520.0/temp))*oh  # in s-1
l_ch4_oh = (2.45e-12*exp(-1775.0/temp))*oh  
l_mcf_oh *= 3600.*24.0*365.0  # in yr-1
l_ch4_oh *= 3600.*24.0*365.0
a_ch4_oh = 1 - 3.9/1000 # fractionation by OH
R_ST = 11237.2e-6 # Standard V-PDP 13C/12C ratio
l_ch4_oh_abs = 528. # Tg yr-1
l_ch4_other_abs = 109. # Tg yr-1
t_ch4 = 9.1 # lifetime methane in yr
l_ch4_other = l_ch4_other_abs / (t_ch4 * (l_ch4_oh_abs +l_ch4_other_abs)) # in yr-1
a_ch4_other = 1 - 19./1000
styear,edyear = 1988,2009

nt = edyear-styear

# Reading in the data
mcf_obs,mcf_obs_e = read_mcf_measurements()
rapid,medium,slow,stock,em0_mcf,prod = read_mcf_emi(os.path.join('EMISSIONS','emissions.dat'))
ch4_obs,ch4_obs_e = read_ch4_measurements()
em0_ch4 = array([550.0]*nt)*1e9
d13c_obs,d13c_obs_e = read_d13C_obs(os.path.join('OBSERVATIONS','d13C_Schaefer.txt'))
em0_d13c = array([-54.1]*nt)
c12_obs, c13_obs, c12_obs_e, c13_obs_e = deltot_to_split(ch4_obs,d13c_obs,ch4_obs_e,d13c_obs_e)
c12_obs_e *= .1
c13_obs_e *= .1
mcf_obs_e *= 1.
em0_c12, em0_c13 = deltot_to_split(em0_ch4, em0_d13c,mass=True)
con_data = array([mcf_obs,ch4_obs,d13c_obs])
con_data_e = array([mcf_obs_e,ch4_obs_e,d13c_obs_e])

mcf_ini = array([117])
ch4_ini = ch4_obs[0]/1.01
d13c_ini = d13c_obs[0]
c12_ini,c13_ini = deltot_to_split(ch4_ini,d13c_ini)
c12_ini,c13_ini = array([c12_ini]),array([c13_ini])
foh_prior  = ones(nt)
fmcf_prior = zeros(nt)
f12_prior  = ones(nt)
f13_prior  = ones(nt)

x_prior = concatenate((foh_prior, mcf_ini, fmcf_prior, \
                        c12_ini, f12_prior, c13_ini, f13_prior))

nstate = len(x_prior)
# Constructing the prior error matrix b
b = zeros((nstate,nstate))
error_oh = 0.02
error_e_mcf = 0.02; error_e_c12h4 = 0.02; error_e_c13h4 = 0.02 # emission errors
error_mcf0 = 0.05; error_c12h40 = 0.05; error_c13h40 = 0.1 # error in initial concentration
corlen_oh = 1.
corlen_em = 1.

b[nt,nt] = (x_prior[nt]*error_mcf0)**2
b[2*nt+1, 2*nt+1] = (x_prior[2*nt+1]*error_c12h40)**2
b[3*nt+2, 3*nt+2] = (x_prior[3*nt+2]*error_c13h40)**2
for i in range(0,nt):
    b[i,i] = error_oh**2
    for j in range(0,i):
        b[i,j] = exp(-(i-j)/corlen_oh)*(error_oh)**2
        b[j,i] = b[i,j]
for i in range(nt+1, 2*nt+1):
    b[i,i] = error_e_mcf**2
for i in range(2*nt+2,3*nt+2): 
    b[i,i] = error_e_c12h4**2
    for j in range(2*nt+2,i):
        b[i,j] = exp(-(i-j)/corlen_em)*(error_e_c12h4)**2
        b[j,i] = b[i,j]
for i in range(3*nt+3,4*nt+3): 
    b[i,i] = error_e_c12h4**2
    for j in range(3*nt+3,i):
        b[i,j] = exp(-(i-j)/corlen_em)*(error_e_c12h4)**2
        b[j,i] = b[i,j]
b_inv = linalg.inv(b)
#     set up preconditioning
L_precon = sqrt(b)
L_adj   = transpose(L_precon)
L_inv   = linalg.inv(L_precon)
xp_prior = state_to_precon(x_prior)

start = time.time()
xp_opt = optimize.fmin_bfgs(calculate_J,xp_prior,calculate_dJdx, gtol=1e-2)
x_opt = precon_to_state(xp_opt)
end   = time.time()
print 10**3*(end-start),'ms'

#plt.figure()
#plt.plot(x_prior)
#plt.plot(x_opt)

mcf_prior = forward_mcf(x_prior)
c12_prior,c13_prior = forward_c12(x_prior),forward_c13(x_prior)
ch4_prior,d13c_prior = split_to_deltot(c12_prior,c13_prior)
con_prior = forward_all(x_prior)
J_ch4_prior  = ( (ch4_prior-ch4_obs)   / ch4_obs_e  )**2
J_d13c_prior = ( (d13c_prior-d13c_obs) / d13c_obs_e )**2
_,_,J_mcf_prior = cost_mcf( con_prior )
_,_,J_c12_prior = cost_c12( con_prior )
_,_,J_c13_prior = cost_c13( con_prior )

mcf_opt = forward_mcf(x_opt)
c12_opt, c13_opt = forward_c12(x_opt), forward_c13(x_opt)
ch4_opt, d13c_opt = split_to_deltot(c12_opt, c13_opt)
con_opt = forward_all(x_opt)
J_ch4_opt  = ( (ch4_opt-ch4_obs)   / ch4_obs_e  )**2
J_d13c_opt = ( (d13c_opt-d13c_obs) / d13c_obs_e )**2
_,_,J_mcf_opt = cost_mcf( con_opt )
_,_,J_c12_opt = cost_c12( con_opt )
_,_,J_c13_opt = cost_c13( con_opt )


f, axarr = plt.subplots(2, sharex=True)
ax1,ax2 = axarr[0],axarr[1]
ax1b,ax2b = ax1.twinx(), ax2.twinx()
ax1.set_title(r'CH$_4$ concentrations and $\delta^{13}$C concentrations:'+' both from\n observations and modelled forward from the prior.')
ax1.set_ylabel(r'CH$_4$ concentration (ppb)')
ax1b.set_ylabel('Cost function')
ax2.set_ylabel(r'$\delta^{13}$C (permil)')
ax2b.set_ylabel('Cost function')
ax2.set_xlabel('Year')
ax1.errorbar(range(styear,edyear),ch4_obs,yerr=ch4_obs_e,fmt = 'o',color = 'red',label=r'CH$_4$, observed' )
ax1.plot( range(styear,edyear),ch4_prior,'r-',label=r'CH$_4$, prior')
ax1.plot( range(styear,edyear),ch4_opt  ,'g-',label=r'CH$_4$, optimized')
ax1b.plot(range(styear,edyear),J_ch4_prior, color = 'red')
ax1b.plot(range(styear,edyear),J_ch4_opt, color = 'green')
ax2.errorbar(range(styear,edyear),d13c_obs, yerr= d13c_obs_e  ,fmt = 'o',color = 'green',label=r'$\delta^{13}$C, observed')
ax2.plot( range(styear,edyear),d13c_prior,'r--',label=r'$\delta^{13}$C, prior')
ax2.plot( range(styear,edyear),d13c_opt  ,'g--',label=r'$\delta^{13}$C, optimized')
ax2b.plot(range(styear,edyear),J_d13c_prior, color = 'red')
ax2b.plot(range(styear,edyear),J_d13c_opt, color = 'green')
ax1.legend(loc='upper left')
ax2.legend(loc='lower right')
plt.savefig('delc_ch4')

fig1, axarr = plt.subplots(2, sharex=True)
ax1,ax2 = axarr[0],axarr[1]
ax1b,ax2b = ax1.twinx(), ax2.twinx()
ax1.set_ylabel(r'$^{12}$CH$_4$ concentrations (ppb)')
ax1b.set_ylabel('Cost function')
ax2.set_ylabel(r'$^{13}$CH$_4$ concentrations (ppb)')
ax2b.set_ylabel('Cost function')
ax2.set_xlabel('Year')
ax1.set_title(r'$^{12}$CH$_4$ and $^{13}$CH$_4$ concentrations:'+' both from observations\n and modelled forward from the prior.')
ax1.errorbar(range(styear,edyear),c12_obs,yerr=c12_obs_e,fmt='o', color = 'green', label = r'$^{12}$CH$_4$ obs')
ax1.plot(range(styear,edyear),c12_prior,'-', color = 'red', label = r'$^{12}$CH$_4$ prior')
ax1.plot(range(styear,edyear),c12_opt,'-', color = 'green', label = r'$^{12}$CH$_4$ optimized')
ax1b.plot(range(styear,edyear),J_c12_prior, color = 'red')
ax1b.plot(range(styear,edyear),J_c12_opt, color = 'green')
ax2.errorbar(range(styear,edyear),c13_obs, yerr=c13_obs_e,fmt= 'o',color = 'green', label = r'$^{13}$CH$_4$ obs')
ax2.plot(range(styear,edyear),c13_prior,'-', color = 'red', label = r'$^{13}$CH$_4$ prior')
ax2.plot(range(styear,edyear),c13_opt,'-', color = 'green', label = r'$^{13}$CH$_4$ optimized')
ax2b.plot(range(styear,edyear),J_c13_prior, color = 'red')
ax2b.plot(range(styear,edyear),J_c13_opt, color = 'green')
ax1.legend(loc = 'upper left')
ax2.legend(loc = 'lower right')
plt.savefig('c12_c13_concentrations')

fig_mcf = plt.figure()
ax1 = fig_mcf.add_subplot(111)
ax1b = ax1.twinx()
ax1.set_xlabel('Year')
ax1b.set_ylabel('Cost function')
ax1.set_ylabel('mcf concentration (ppb)')
ax1.set_title(r'mcf concentrations:'+' both from observations\n and modelled forward from the prior.')
ax1.errorbar(range(styear,edyear),mcf_obs, yerr = mcf_obs_e,fmt='o',color='gray',label = 'mcf observations')
ax1.plot(range(styear,edyear),mcf_prior,'-',color='red', label = 'mcf prior')
ax1.plot(range(styear,edyear),mcf_opt  ,'-',color='green', label = 'mcf optimized')
ax1b.plot(range(styear,edyear),J_mcf_prior, color = 'red')
ax1b.plot(range(styear,edyear),J_mcf_opt, color = 'green')
ax1.legend(loc='best')
plt.savefig('mcf_concentrations')




























