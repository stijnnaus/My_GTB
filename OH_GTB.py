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
    
    J_obs = J_mcf + J_c12 + J_c13 # mismatch with obs
    J_pri = sum(dot(b_inv, ( x - x_prior )**2)) # prior
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
    dfoh_c12,dc12i,dfc12 = adjoint_model_c12( dep_c12, foh, c12_save )
    dfoh_c13,dc13i,dfc13 = adjoint_model_c13( dep_c13, foh, c13_save )
    dfoh = dfoh_mcf + dfoh_c12 + dfoh_c13
    
    dJdx_obs = np.concatenate((dfoh, dmcfi, dfmcf, \
                                dc12i, dfc12, dc13i, dfc13))
    dJdx_pri = dot(b_inv, (x-x_prior))
    dJdx = dJdx_obs + dJdx_pri
    dJdxp = dot( L_adj, dJdx )
    print 'Cost function deriv:',max(dJdxp)
    return dJdxp

def adjoint_model_mcf( dep, foh, mcf_save ):
    pulse_mcf = adj_obs_oper( dep )
    dmcf,dfmcf,dfoh = zeros(nt),zeros(nt),zeros(nt)
    dmcfi = 0.
    rapidc = rapid / conv_mcf
    
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
        
        dfoh[i] = - a_ch4_oh * l_ch4_oh * c13_save[i] * dc13i
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
    
    dmcfs = zeros(nt); dmcfi = dmcf0
    rapidc = rapid/conv_mcf
    
    for year in range(styear,edyear):
        i  = year - styear
        ie = year - 1951
        
        # Emissions
        dmcfi -= 0.75 * rapidc[ie] * dfmcf[i]
        if year > styear: dmcfi -= 0.25 * rapidc[ie-1] * dfmcf[i-1]
        for yearb in range(year-1,year-11,-1):
            i2  = yearb - styear
            ie2 = yearb - 1951
            if yearb >= styear: dmcfi += 0.1*dfmcf[i2]*rapidc[ie2]
                
#        # Chemistry
        dmcfi = dmcfi * ( 1. - l_mcf_ocean - l_mcf_strat - l_mcf_oh * foh[i] ) - \
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
                dfoh[i] * c13_save[i] * l_ch4_oh * a_ch4_oh
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

def split_to_deltot(c12, c13, c12_e = [None], c13_e = [None], mass=False):
    '''
    Converts split c12 and c13 values (emission/concentration)  
    to a delta 13C value and a total CH4 mass.
    '''
    
    ch4 = c12 + c13
    if mass:
        c12 /= conv_ch4
        c13 /= conv_c13
    R_sample = c13 / c12
    d13c = 1000 * (( R_sample / R_ST ) - 1)
    
    if c12_e[0] != None and c13_e[0] != None:
        ch4_e = sqrt( c12_e**2 + c13_e**2 )
        if mass:
            c12_e /= conv_ch4
            c13_e /= conv_c13
        d13c_e = sqrt( 1000 * ( ( - (c13/c12**2) / R_ST )**2 * c12_e**2 + \
                                ( (1/c12) / R_ST )**2 * c13_e**2 ) )
        return ch4, d13c, ch4_e, d13c_e
    
    return ch4, d13c

def deltot_to_split(ch4, delc, ch4_e = [None], delc_e = [None],mass=False):
    '''
    Converts a (list of) d13C and total CH4 values to c12 and c13.
    If abso = True, then the total CH4 input is in mass (eg emissions), and a 
    correction is needed, since 1 mol c12 has a different mass than 1 mol
    c13.
    '''
    if type(delc) == float:
        q = R_ST * ( delc/1000. + 1 )
        c12 = ch4 / (1+q)
        c13 = q * c12
        return c12, c13
        
    ch4, delc = array(ch4), array(delc)/1000.
    q = array( R_ST * (delc+1) )
    c12 = ch4 / (q+1)
    c13 = q*c12
    if mass:
        c13 *= (conv_c13 / conv_ch4)
    
    if delc_e[0] != None and ch4_e[0] != None:
        ch4_e, delc_e = array(ch4_e), array(delc_e)/1000.
        c12_e = sqrt((   ch4_e / (1+q) )**2 + \
                    ( delc_e * ch4 * R_ST / (1+q)**2 )**2)
        c13_e = sqrt(( q*ch4_e / (1+q) )**2 + \
                    ( delc_e * ch4 * R_ST * (1/(1+q) - q/(1+q)**2) )**2)
        if mass:
            c13_e *= (conv_c13 / conv_ch4)
        return c12, c13, c12_e, c13_e
        
    return c12, c13

def unpack(x):
    '''
    Unpacks the state vector. Returns the respective components.
    '''
    foh = x[:nt]
    mcfi,c12i,c13i = x[nt],x[2*nt+1],x[3*nt+2]
    fst,fc12,fc13 = x[nt+1:2*nt+1],x[2*nt+2:3*nt+2],x[3*nt+3:4*nt+3]
    fsl = x[4*nt+3:5*nt+3]
    
    return foh, mcfi, fmcf, c12i, fc12, c13i, fc13    

# setup model variables, emissions, etc:
exp_name = '_Loose_Prior'
m = 5.e18
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

fec,femcf = 1.,1. # Reduction of the error
mcf_obs_e *= femcf
ch4_obs_e *= fec
d13c_obs_e *= fec

c12_obs, c13_obs, c12_obs_e, c13_obs_e = deltot_to_split(ch4_obs,d13c_obs,ch4_obs_e,d13c_obs_e)

em0_c12, em0_c13 = deltot_to_split(em0_ch4, em0_d13c,mass=True)
con_data = array([mcf_obs,ch4_obs,d13c_obs])
con_data_e = array([mcf_obs_e,ch4_obs_e,d13c_obs_e])

mcf0_prior = array([117])
ch40_prior = ch4_obs[0] / 1.01
d13c0_prior = d13c_obs[0]
c120_prior,c130_prior = deltot_to_split(ch40_prior,d13c0_prior)
c120_prior,c130_prior = array([c120_prior]),array([c130_prior])
foh_prior  = ones(nt)
fmcf_prior = zeros(nt)
f12_prior  = ones(nt)
f13_prior  = ones(nt)

x_prior = concatenate((foh_prior, mcf0_prior, fmcf_prior, \
                        c120_prior, f12_prior, c130_prior, f13_prior))

nstate = len(x_prior)
# Constructing the prior error matrix b
b = zeros((nstate,nstate))
error_oh = .2
error_e_mcf = 1.2; error_e_c12 = 1.2; error_e_c13 = 1.2 # emission errors
error_mcf0 = 1.2; error_c120 = 1.2; error_c130 = 1.4 # error in initial concentration
corlen_oh = 1. 
corlen_em = 1.

b[nt,nt] = (x_prior[nt]*error_mcf0)**2
b[2*nt+1, 2*nt+1] = (x_prior[2*nt+1]*error_c120)**2
b[3*nt+2, 3*nt+2] = (x_prior[3*nt+2]*error_c130)**2
for i in range(0,nt):
    b[i,i] = error_oh**2
    for j in range(0,i):
        b[i,j] = exp(-(i-j)/corlen_oh)*(error_oh)**2
        b[j,i] = b[i,j]
for i in range(nt+1, 2*nt+1):
    b[i,i] = error_e_mcf**2
for i in range(2*nt+2,3*nt+2): 
    b[i,i] = error_e_c12**2
    for j in range(2*nt+2,i):
        b[i,j] = exp(-(i-j)/corlen_em)*(error_e_c12)**2
        b[j,i] = b[i,j]
for i in range(3*nt+3,4*nt+3): 
    b[i,i] = error_e_c13**2
    for j in range(3*nt+3,i):
        b[i,j] = exp(-(i-j)/corlen_em)*(error_e_c13)**2
        b[j,i] = b[i,j]
        

b_inv = linalg.inv(b)
#     set up preconditioning
L_precon = sqrt(b)
L_adj   = transpose(L_precon)
L_inv   = linalg.inv(L_precon)
xp_prior = state_to_precon(x_prior)

start = time.time()
xp_opt = optimize.fmin_bfgs(calculate_J,xp_prior,calculate_dJdx, gtol=1e-4)
x_opt = precon_to_state(xp_opt)
end   = time.time()
print 10**3*(end-start),'ms'

#plt.figure()
#plt.plot(x_prior)
#plt.plot(x_opt)

mcf_prior = forward_mcf(x_prior)
c12_prior,c13_prior = forward_c12(x_prior),forward_c13(x_prior)

ch4_prior,d13c_prior = split_to_deltot( c12_prior ,c13_prior )
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

figSize = (10,10)
f, axarr = plt.subplots(2, sharex=True, figsize = figSize)
ax1,ax2 = axarr[0],axarr[1]
ax1b,ax2b = ax1.twinx(), ax2.twinx()
ax1.set_title(r'CH$_4$ concentrations and $\delta^{13}$C concentrations:'+' both from\n observations and modelled forward from the prior.')
ax1.set_ylabel(r'CH$_4$ concentration (ppb)')
ax1b.set_ylabel('Cost function')
ax2.set_ylabel(r'$\delta^{13}$C (permil)')
ax2b.set_ylabel('Cost function')
ax2.set_xlabel('Year')
ax1.errorbar(range(styear,edyear),ch4_obs,yerr=ch4_obs_e,fmt = 'o',color = 'green',label=r'CH$_4$, observed' )
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
plt.savefig('d13C_CH4_concentrations'+exp_name)

fig1, axarr = plt.subplots(2, sharex=True,figsize = figSize)
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
plt.savefig('C12_C13_concentrations'+exp_name)

fig_mcf = plt.figure(figsize = figSize)
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
plt.savefig('MCF_concentrations'+exp_name)

# Plotting prior state vs optimized state
foh_opt, mcf0_opt, fmcf_opt = x_opt[:nt], x_opt[nt], x_opt[nt+1:2*nt+1]
c120_opt, f12_opt = x_opt[2*nt+1], x_opt[2*nt+2:3*nt+2]
c130_opt, f13_opt = x_opt[3*nt+2], x_opt[3*nt+3:4*nt+4]

oh_prior, oh_opt = foh_prior*oh/1.e6, foh_opt*oh/1.e6
errors_oh = np.array([error_oh]*nt)*oh_prior
emcf_prior, emcf_opt = em0_mcf + mcf_shift(fmcf_prior), em0_mcf + mcf_shift(fmcf_opt)
errors_e_mcf = np.array([error_e_mcf]*nt)*emcf_prior
ec12_prior, ec12_opt = f12_prior * em0_c12, f12_opt * em0_c12
ec13_prior, ec13_opt = f13_prior * em0_c13, f13_opt * em0_c13
ec12_error, ec13_error = error_e_c12 * em0_c12, error_e_c13 * em0_c13
ech4_prior, ed13c_prior, ech4_error, ed13c_error = split_to_deltot( ec12_prior, ec13_prior, ec12_error, ec13_error, mass=True )
ech4_opt  , ed13c_opt   = split_to_deltot( ec12_opt  , ec13_opt  , mass=True )

fig = plt.figure(figsize = figSize)
ax1 = fig.add_subplot(111)
ax1.set_title('Prior and optimized global mean OH concentrations')
ax1.set_xlabel('Year')
ax1.set_ylabel(r'OH concentration ($10^6$ molec cm$^{-3}$)')
ax1.plot( range(styear,edyear), oh_prior, 'o-', color = 'red',   label = 'Prior'     )
ax1.plot( range(styear,edyear), oh_opt  , 'o-', color = 'green', label = 'Optimized' )
ax1.legend(loc='best')
plt.savefig('OH_field_prior_opt'+exp_name)

fig = plt.figure(figsize = figSize)
ax1 = fig.add_subplot(111)
ax1.set_title('Prior and optimized global mean MCF emissions')
ax1.set_xlabel('Year')
ax1.set_ylabel('MCF emissions (Gg/yr)')
ax1.plot( range(styear,edyear), emcf_prior/1.e6, 'o-', color = 'red',   label = 'Prior'     )
ax1.plot( range(styear,edyear), emcf_opt/1.e6  , 'o-', color = 'green', label = 'Optimized' )
ax1.legend(loc='best')
plt.savefig('emi_MCF_prior_opt'+exp_name)

fig = plt.figure(figsize = figSize)
ax1 = fig.add_subplot(211)
ax1.set_title(r'Prior and optimized global mean CH$_4$ emissions')
ax1.set_xlabel('Year')
ax1.set_ylabel('CH$_4$ emissions (Tg/yr)')
ax1.plot( range(styear,edyear), ech4_prior/1.e9, 'o-', color = 'red',   label = 'Prior'     )
ax1.plot( range(styear,edyear), ech4_opt/1.e9  , 'o-', color = 'green', label = 'Optimized' )
ax1.legend(loc='best')
ax2 = fig.add_subplot(212)
ax2.set_title(r'Prior and optimized $\delta^{13}$C global mean CH$_4$ emissions')
ax2.set_xlabel('Year')
ax2.set_ylabel('$\delta^{13}$C (permil)')
ax2.plot( range(styear,edyear), ed13c_prior, 'o-', color = 'red',   label = 'Prior'     )
ax2.plot( range(styear,edyear), ed13c_opt  , 'o-', color = 'green', label = 'Optimized' )
ax2.legend(loc='best')
plt.savefig('emi_CH4_d13C_prior_opt'+exp_name)


rel_dev = (x_opt - x_prior)
fig = plt.figure(figsize = figSize)
ax1 = fig.add_subplot(111)
ax1.set_title('Relative deviations from prior')
ax1.set_xlabel('Year')
ax1.set_ylabel('Relative deviation (%)')
ax1.plot( range(styear,edyear), rel_dev[:nt], 'o-', color = 'blue', label = 'OH')
ax1.plot( range(styear,edyear), rel_dev[nt+1:2*nt+1], 'o-', color = 'green', label = 'MCF' )
ax1.plot( range(styear,edyear), rel_dev[2*nt+2:3*nt+2], 'o-', color = 'red', label = r'$^{12}$CH$_4$' )
ax1.plot( range(styear,edyear), rel_dev[3*nt+3:4*nt+3], 'o-', color = 'maroon', label = r'$^{13}$CH$_4$' )
ax1.legend(loc='best')
plt.savefig('rel_dev_from_prior'+exp_name)


J_prior = dot(b_inv, ( x_opt - x_prior )**2) # prior
J_prior_ch4 = (( ech4_prior - ech4_opt ) / ech4_error)**2
J_prior_d13c = (( ed13c_prior - ed13c_opt ) / ed13c_error)**2

fig_J = plt.figure(figsize = figSize)
ax1 = fig_J.add_subplot(212)
ax2 = fig_J.add_subplot(211)
ax1.set_title('Second part cost function')
ax2.set_title('First part cost function')
ax2.set_xlabel('Year')
ax1.set_ylabel('Cost function')
ax2.set_ylabel('Cost function')
[range(styear,edyear),]
ax1.plot( range(styear,edyear), J_prior[:nt], 'o-', color = 'blue', label = 'OH' )
ax1.plot( range(styear,edyear), J_prior[nt+1:2*nt+1], 'o-', color = 'green', label = 'MCF' )
ax1.plot( range(styear,edyear), J_prior[2*nt+2:3*nt+2], 'o-', color = 'red', label = r'$^{12}$CH$_4$' )
ax1.plot( range(styear,edyear), J_prior[3*nt+3:4*nt+3], 'o-', color = 'maroon', label = r'$^{13}$CH$_4$' )
ax2.plot( range(styear,edyear), J_mcf_opt, 'o-', color = 'green', label = 'MCF' )
ax2.plot( range(styear,edyear), J_c12_opt, 'o-', color = 'red', label = r'$^{12}$CH$_4$' )
ax2.plot( range(styear,edyear), J_c13_opt, 'o-', color = 'maroon', label = r'$^{13}$CH$_4$' )
ax1.legend(loc='best')
ax2.legend(loc='best')
plt.savefig('cost_functions_state'+exp_name)



plt.figure()
plt.title('Second part of cost function for '+r'CH$_4$ and $\delta^{13}$C')
plt.xlabel('Year')
plt.ylabel('Cost function')
plt.plot( range(styear,edyear), J_prior_ch4 ,'o-', color = 'red', label = r'CH$_4$')
plt.plot( range(styear,edyear), J_prior_d13c,'o-', color = 'maroon', label = r'$\delta^{13}$C')
plt.legend(loc = 'best')
plt.savefig('cost_function_ch4_d13c'+exp_name)







mcfff = mcf_obs
























