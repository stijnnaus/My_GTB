# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 11:36:16 2016

@author: Stijn
"""

import sys
import os
import time
import numpy as np
from numpy import array
from numpy import sqrt
from pylab import *
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from copy import *
from numpy import linalg
from scipy import optimize

def calculate_J(xp):
    x = precon_to_state(xp)
    mcf = forward_mcf(x)
    ch4, r13, _ = forward_ch4(x)
    _, J_mcf, _ = cost_mcf(mcf)
    _, J_ch4, _ = cost_ch4(ch4)
    _, J_r13, _ = cost_r13(r13)
    J_obs = J_mcf + J_ch4 + J_r13
    _, J_pri, _ = cost_bg(x) # mismatch with prior
    J_tot = .5 * (J_pri+J_obs)
    #print 'Cost observations  :',J_obs*red
    #print 'Cost background    :',J_pri*red
    #print 'Cost function value:',J_tot*red
    return J_tot*red
    
def calculate_dJdx(xp):
    x = precon_to_state(xp)
    _, ch4i, r13i, foh, _, _, _, fch4, r13e = unpack(x)
    mcf_sv = forward_mcf(x)
    ch4sv, r13sv, c13sv = forward_ch4(x)
    
    dep_mcf, _, _ = cost_mcf(mcf_sv)
    dep_ch4, _, _ = cost_ch4(ch4sv)
    dep_r13, _, _ = cost_r13(r13sv)
    
    dmcf0, dfoh_mcf, dfst, dfsl, dfme = adjoint_model_mcf(dep_mcf, foh, mcf_sv)
    dch4i, dr13i, dfoh_ch4, dfch4, dr13e = adjoint_model_ch4(dep_ch4, dep_r13, 
                                                ch4i, r13i, foh, fch4, r13e,
                                                ch4sv, c13sv)
    dfoh = dfoh_mcf + dfoh_ch4
    
    dJdx_obs = np.concatenate((dmcf0, dch4i, dr13i, dfoh, \
                                dfst, dfsl, dfme, dfch4, dr13e))
    dJdx_pri, _, _ = cost_bg(x)
    dJdx = dJdx_obs + dJdx_pri
    dJdxp = np.dot(L_adj, dJdx)
    #print 'Cost function deriv:',max(dJdxp)*red
    return dJdxp*red
    
def cost_bg(x):
    '''The cost and deriv from mismatch with the prior.'''
    cost_list_bg = np.dot(b_inv, ( x - x_pri )**2)
    cost_bg = sum(cost_list_bg)
    deriv_bg = np.dot(b_inv, ( x - x_pri ))
    return deriv_bg, cost_bg, cost_list_bg
    
def cost_mcf(mcf):
    dif = (mcf - mcf_obs)
    dep = dif / mcf_obs_e**2
    cost_list = dif*dep
    cost = sum(cost_list)
    return dep,cost,cost_list
    
def cost_ch4(ch4):
    dif = (ch4 - ch4_obs)
    dep = dif / ch4_obs_e**2
    cost_list = dif*dep
    cost = sum(cost_list)
    return dep,cost,cost_list

def cost_r13(r13m):
    dif = (r13m - r13_obs)
    dep = dif / r13_obs_e**2
    cost_list = dif*dep
    cost = sum(cost_list)
    return dep, cost, cost_list

def adjoint_model_mcf( dep, foh, mcf_save ):
    pulse_mcf = adj_oper_av( dep )
    admcf,adfoh = np.zeros(nt),np.zeros(nt)
    adfst,adfsl,adfme = np.zeros(nt),np.zeros(nt),np.zeros(nt)
    adshift = np.zeros(nt)
    admcfi = 0.
    rapidc = rapid / conv_mcf
    for iyear in range(edyear-1,styear-1,-1):
        i  = iyear-styear
        ie = iyear-1951
        for n in range(0,nstep):
            # Add adjoint pulse
            admcfi  += pulse_mcf[i][n]
            # Chemistry
            adfoh[i] -= l_mcf_ohf * mcf_save[i] * admcfi
            admcfi   = admcfi *  (1. - foh[i] * l_mcf_ohf - l_mcf_stratf - l_mcf_oceanf)
            # Emission shift
            adshift[i] += admcfi*dt
        # Emissions stock
        adfst[i] -= 0.75 * rapidc[ie] * adshift[i]
        if (iyear + 1) < edyear: adfst[i] -= 0.25 * rapidc[ie] * adshift[i+1]
        for yearb in range(iyear+1 , iyear+11):
            i2  = yearb - styear
            if yearb < edyear: adfst[i] += 0.1 * rapidc[ie] * adshift[i2]
        # Emissions slow
        adfsl[i] -= 0.75 * rapidc[ie] * adshift[i]
        if (iyear + 2) < edyear: adfsl[i] += 0.75 * rapidc[ie] * adshift[i+2]
        admcf[i] = admcfi
        # Emissions medium
        adfme[i] -= 0.5 * rapidc[ie] * adshift[i]
        if (iyear+1) < edyear: adfme[i] += 0.5 * rapidc[ie] * adshift[i+1]
    adj_mcf = [array([admcfi]),adfoh,adfst,adfsl,adfme]
    return adj_mcf
    
def adjoint_model_ch4(dep_ch4, dep_r13, ch4i_sv, r13i_sv, foh_sv, fch4_sv, r13e_sv, 
                      ch4sv, c13sv):
    pulse_ch4_av, pulse_c13_av = adj_oper_ch4( dep_ch4, dep_r13, ch4sv, c13sv )
    pulse_ch4, pulse_c13 = adj_oper_av(pulse_ch4_av), adj_oper_av(pulse_c13_av)
    dfch4 = np.zeros(nt)
    dr13e = np.zeros(nt)
    dfoh = np.zeros(nt)
    dc13i = 0.
    dch4i = 0.
    em_ch4 = em0_ch4/conv_ch4*dt
    for iyear in range(edyear-1, styear-1, -1):
        i = iyear - styear
        for n in range(nstep):
            dch4i += pulse_ch4[i][n]
            dc13i += pulse_c13[i][n]
            # Chemistry
            dfoh[i] -= ch4sv[i]*l_ch4_ohf*dch4i
            dch4i = dch4i * (1 - foh_sv[i]*l_ch4_ohf - l_ch4_otherf)
            dfoh[i] -= a_ch4_oh*l_ch4_ohf*c13sv[i]*dc13i
            dc13i = dc13i * (1 - foh_sv[i]*a_ch4_oh*l_ch4_ohf - a_ch4_other*l_ch4_otherf)
            # Emissions
            dfch4[i] += em_ch4[i] * (dch4i + r13e_sv[i]*dc13i)
            dr13e[i] += fch4_sv[i] * em_ch4[i] * dc13i
    # Initialization
    dch4i = dch4i + r13i_sv * dc13i
    dr13i = ch4i_sv * dc13i
    adj_ch4 = [array([dch4i]), array([dr13i]), dfoh, dfch4, dr13e]
    return adj_ch4

def forward_tl_mcf(x, foh, mcf_save):
    dmcf0, _, _, dfoh, dfst, dfsl, dfme, _, _ = unpack(x)
    dmcfs = np.zeros((nt,nstep))
    dmcfi = dmcf0
    rapidc = rapid/conv_mcf
    for year in years:
        i  = year - styear
        ie = year - 1951
        # Calculate emission shift from fstock ...
        dshift = - 0.75 * rapidc[ie] * dfst[i]
        if i >= 1: dshift -= 0.25 * rapidc[ie-1] * dfst[i-1]
        for yearb in range(year-1,year-11,-1):
            i2  = yearb - styear
            ie2 = yearb - 1951
            if yearb >= styear: dshift += 0.1*dfst[i2]*rapidc[ie2]
        # ... from fslow ...
        dshift -= 0.75 * rapidc[ie] * dfsl[i]
        if i >= 2: dshift += 0.75 * rapidc[ie-2] * dfsl[i-2]
        # ... and from fmedium.
        dshift -= 0.5 * rapidc[ie] * dfme[i]
        if i >= 1: dshift += 0.5 * rapidc[ie-1] * dfme[i-1]
        dshiftf = dshift*dt
        # Emission-chemistry interaction
        for n in range(nstep):
            # Add emission
            dmcfi += dshiftf
            # Chemistry
            dmcfi = dmcfi * ( 1. - l_mcf_oceanf - l_mcf_stratf - l_mcf_ohf * foh[i] ) - \
                    dfoh[i] * mcf_save[i] * l_mcf_ohf
            dmcfs[i][n] = dmcfi
    return obs_oper_av(dmcfs)
    
def forward_tl_ch4(x, ch4i_sv, r13i_sv, foh_sv, fch4_sv, r13e_sv, ch4sv, c13sv):
    _,dch4i, dr13i, dfoh, _, _, _, dfch4, dr13e = unpack(x)
    dch4s, dc13s = np.zeros((nt, nstep)), np.zeros((nt, nstep))
    # Initialization
    dch4 = dch4i
    dc13 = dr13i*ch4i_sv + r13i_sv*dch4i
    # Emissions
    em0_ch4p = em0_ch4/conv_ch4*dt
    em_ch4 = fch4_sv * em0_ch4p
    dem_ch4 = dfch4 * em0_ch4p
    dem_c13 = em_ch4*dr13e + dem_ch4*r13e_sv
    for year in years:
        i = year - styear
        for n in range(nstep):
            dch4 += dem_ch4[i]
            dc13 += dem_c13[i]
            # Chemistry
            dch4 = dch4 * (1 - l_ch4_ohf*foh_sv[i] - l_ch4_otherf) \
                    - ch4sv[i]*l_ch4_ohf*dfoh[i]
            dc13 = dc13 * (1 - a_ch4_other*l_ch4_otherf - a_ch4_oh*l_ch4_ohf*foh_sv[i])\
                    - c13sv[i]*a_ch4_oh*l_ch4_ohf*dfoh[i]
            dc13s[i][n] = dc13
            dch4s[i][n] = dch4
    dch4_av, dc13_av = obs_oper_av(dch4s), obs_oper_av(dc13s)
    dch4_ob, dr13_ob = obs_oper_ch4_tl(dch4_av, dc13_av, ch4sv, c13sv)
    return dch4_ob, dr13_ob, dc13_av
    
def forward_all(x):
    mcf,ch4,r13,c13 = forward_mcf(x),forward_ch4(x)
    return mcf, ch4, r13, c13

def forward_mcf(x,lt=False):
    '''
    Forward MCF model. 
    lt: If True, it also returns the MCF lifetimes set by OH
    '''
    mcf0, _, _, foh, fst, fsl, fme, _, _ = unpack(x)
    em = em0_mcf + mcf_shift(fst, fsl, fme)
    em /= conv_mcf
    mcfs = np.zeros((nt,nstep))
    mcf = mcf0
    for year in years:
        i = year - styear
        emf = em[i]*dt
        for n in range(nstep):
            mcf += emf
            mcf = mcf * ( 1 - l_mcf_ohf * foh[i] - l_mcf_oceanf - l_mcf_stratf )
            mcfs[i][n] = mcf
    if lt==True: return obs_oper_av(mcfs),tau_oh,tau
    return obs_oper_av(mcfs)

def forward_ch4(x,lossoh=False):
    _, ch4i, r13i, foh, _, _, _, fch4, r13e = unpack(x)
    ch4s, c13s = np.zeros((nt,nstep)), np.zeros((nt,nstep))
    # Initialization
    ch4 = ch4i
    c13 = r13i*ch4i
    # Emissions
    em_ch4 = fch4 * (em0_ch4 / conv_ch4) * dt
    em_c13 = r13e * em_ch4
    for year in years:
        i = year - styear
        for n in range(nstep):
            ch4 += em_ch4[i]
            c13 += em_c13[i]
            # Chemistry
            ch4 = ch4 * ( 1 - l_ch4_otherf - l_ch4_ohf*foh[i])
            c13 = c13 * ( 1 - a_ch4_other*l_ch4_otherf - a_ch4_oh*l_ch4_ohf*foh[i] )
            ch4s[i][n] = ch4
            c13s[i][n] = c13
    ch4s_av, c13s_av = obs_oper_av(ch4s), obs_oper_av(c13s)
    ch4s_n, r13s_n = obs_oper_ch4(ch4s_av, c13s_av)
    if lossoh: return ch4s_n, r13s_n, c13s_av, loh
    return ch4s_n, r13s_n, c13s_av

def mcf_shiftx(x):
    _,_,_,_,fst,fsl,fme,_,_ = unpack(x)
    return mcf_shift(fst,fsl,fme)

def mcf_shift(fst, fsl, fme):
    '''
    Computes the shift in emissions that results from repartitioning of the production
    fst: the fraction of rapid emissions that is instead moved to the stockpile
    fsl: the fraction of rapid emissions that is instead moved to slow emissions
    ''' 
    shifts = []
    for year in years:
        i = year - styear # Index in fsl and fst
        ie = year - 1951  # Index in emissions
        # Stock
        shift = - 0.75 * rapid[ie] * fst[i]
        if i >= 1: shift -= 0.25 * rapid[ie-1] * fst[i-1]
        for yearb in range(year-1,year-11,-1):
            i2 = yearb - styear  
            ie2 = yearb - 1951
            if i2 >= 0: shift += 0.1 * fst[i2] * rapid[ie2]
        # Slow
        shift -= 0.75 * rapid[ie] * fsl[i]
        if i >= 2: shift += 0.75 * rapid[ie-2] * fsl[i-2]
        # Medium
        shift -= 0.5 * rapid[ie] * fme[i]
        if i >= 1: shift += 0.5 * rapid[ie-1] * fme[i-1]
        shifts.append(shift)
    return array(shifts)
    
def obs_oper_av(con):
    '''Converts from nsteps per year to yearly averages. '''
    means = np.zeros(nt)
    for i,c in enumerate(con):
        means[i] = sum(c)*dt
    return means
    
def adj_oper_av(means):
    '''Converts a yearly average to pulses per nstep.'''
    pulses = []
    for mean in means:
        pulse = [mean*dt for i in range(nstep)]
        pulses.append(array(pulse))
    return pulses
    
def obs_oper_ch4(ch4, c13):
    ''' 
    Converts CH4 and 13CH4 mixing ratios to a CH4 mixing ratio and
    a 13CH4/CH4 ratio r13.
    '''
    ch4 = ch4
    r13 = c13/ch4
    return ch4, r13

def obs_oper_ch4_tl(dch4, dc13, ch4sv, c13sv):
    ''' Tangent linear version of ch4 obs operator '''
    dch4 = dch4
    dr13 = dc13/ch4sv - (c13sv/ch4sv**2) * dch4
    return dch4, dr13

def adj_oper_ch4(dep_ch4, dep_r13, ch4sv, c13sv):
    ''' Converts departures in CH4 and r13 to departures in CH4 and 13CH4. '''
    dep_ch4 = dep_ch4 - (c13sv/ch4sv**2) * dep_r13
    dep_c13 = dep_r13 / ch4sv
    return dep_ch4, dep_c13
    
def state_to_precon(x):
    ''' Convert the state and derivative to the preconditioned space. '''
    #return x
    return np.dot( L_inv, (x - x_pri) ) 

def precon_to_state(xp):
    ''' Convert the preconditioned state to the original space. '''
    #return xp
    return np.dot( L_precon, xp ) + x_pri

def d13c_to_r13(d13c, d13c_e=None):
    ''' Converts d13C values to r13 ratios. Optionally converts errors. '''
    d13c = np.asarray(d13c)
    q = R_ST * (d13c/1000.+1)
    r13 = q/(1+q)
    if d13c_e==None: return r13
    d13c_e = np.asarray(d13c_e)
    r13_e = (R_ST / (1000.*(1+q)**2)) * d13c_e
    return r13, r13_e

def r13_to_d13c(r13, r13_e=None):
    ''' Converts r13 ratios to d13c values. Optionally converts errors. '''
    r13 = np.asarray(r13)
    d13c = 1000. * (r13/(R_ST*(1-r13)) - 1)
    if r13_e==None:
        return d13c
    r13_e = np.asarray(r13_e)
    d13c_e = (1000 / (R_ST*(1-r13**2))) * r13_e
    return d13c, d13c_e

def unpack(x):
    ''' Unpacks the state vector. Returns the respective components. '''
    mcfi,ch4i,d13ci = x[0],x[1],x[2]
    foh = x[3:nt+3]
    fstock,fslow,fmed = x[nt+3:2*nt+3],x[2*nt+3:3*nt+3],x[3*nt+3:4*nt+3]
    fch4,fr13 = x[4*nt+3:5*nt+3],x[5*nt+3:6*nt+3]    
    return mcfi, ch4i, d13ci, foh, fstock, fslow, fmed, fch4, fr13

# Tuneable parameters
dataset = 'noaa'
exp_name = 'normal'+'_'+dataset
header_p1 = '#\n'
nstep = 400
temp = 272.0  # Kelvin        
oh = .9*1e6  # molecules/cm3
styear,edyear = 1992,2015
years = np.arange(styear,edyear)
fyears = np.arange(1951,edyear) # years for which I have MCF production
red = 1e-3
save_fig = False # If true it save the figures that are produced
write_data = False # If true writes opt results to output file

# Constants
dt = 1./nstep
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
l_ch4_otherf = l_ch4_other * dt # loss at increased frequency
l_ch4_ohf = l_ch4_oh * dt
l_mcf_ohf = l_mcf_oh * dt
l_mcf_oceanf = l_mcf_ocean * dt
l_mcf_stratf = l_mcf_strat * dt
nt = edyear-styear

# Reading in the emission data and observations
_, mcf_noaa = read_glob_mean(os.path.join('OBSERVATIONS', 'mcf_noaa_glob.txt'), styear, edyear)
_, mcf_agage = read_glob_mean(os.path.join('OBSERVATIONS', 'mcf_agage_glob.txt'), styear, edyear)
_, ch4_noaa, ch4_noaa_e = read_glob_mean(os.path.join('OBSERVATIONS', 'ch4_noaa_glob.txt'), styear, edyear, errors=True)
_, ch4_agage = read_glob_mean(os.path.join('OBSERVATIONS', 'ch4_agage_glob.txt'), styear, edyear)
mcf_obs, mcf_obs_e = read_mcf_measurements()
mcf_obs_e = mcf_obs_e[5:]
yrs_mcf, mcf_obs = read_glob_mean(os.path.join('OBSERVATIONS', 'mcf_'+dataset+'_glob.txt'), styear, edyear)
for yr in range(2008,edyear):
    i = yr - 2008
    mcf_obs_e = np.append(mcf_obs_e,mcf_obs_e[-1]*.9**i)
rapid,medium,slow,stock,em0_mcf,prod = read_mcf_emi(os.path.join('EMISSIONS','emissions.dat'))
rapid,medium,slow,stock,em0_mcf,prod = extend_mcf_emi(rapid,medium,slow,stock,em0_mcf,edyear)
#ch4_obs,ch4_obs_e = read_ch4_measurements()
yrs_ch4, ch4_obs = read_glob_mean(os.path.join('OBSERVATIONS', 'ch4_'+dataset+'_glob.txt'), styear, edyear)
#ch4_obs_e = array([3.0]*nt)
ch4_obs_e = ch4_noaa_e
em0_ch4 = array([590.0]*nt)*1e9
d13c_obs,d13c_obs_e = read_d13C_obs(os.path.join('OBSERVATIONS','d13C_Schaefer.txt'))
d13c_obs, d13c_obs_e = d13c_obs[4:], d13c_obs_e[4:]
em0_d13c = array([-53.]*nt)
r13_obs, r13_obs_e = d13c_to_r13(d13c_obs, d13c_obs_e)
r13e0 = d13c_to_r13(em0_d13c)

fec,femcf = 1.,1. # Reduction of the error
mcf_obs_e *= femcf
ch4_obs_e *= fec
d13c_obs_e *= fec
r13_obs_e *= fec

# The prior
mcf0_pri = array([117.])
ch40_pri = array([1730.])
r130_pri = array([r13_obs[0]])
foh_pri = np.ones(nt)
fsl_pri = np.zeros(nt)
fst_pri = np.zeros(nt)
fme_pri = np.zeros(nt)
fch4_pri = np.ones(nt)
r13e_pri = r13e0

x_pri = np.concatenate((mcf0_pri, ch40_pri, r130_pri, foh_pri, \
                       fst_pri,  fsl_pri, fme_pri,  fch4_pri,  r13e_pri))

pri_e_red = .3

nstate = len(x_pri)
# Constructing the prior error matrix b
b = np.zeros((nstate,nstate))
foh_e = .03*pri_e_red # error in initial oh fields
fst_e = .03*pri_e_red; fsl_e = .03*pri_e_red; fme_e = .03*pri_e_red   # mcf emission errors
fch4_e = .15*pri_e_red; ed13c_e = .5*pri_e_red # ch4 (%) & d13c (perm) emission errors
_, r13e_e = d13c_to_r13(em0_d13c[0], ed13c_e) # resulting error in r13e
mcfi_e = 5.; ch4i_e = 5.; d13ci_e = 1. # error in initial values
_, r13i_e = d13c_to_r13(d13c_obs[0], d13ci_e) # resulting error in r13i
corlen_oh = .5 
corlen_em = 2.

b[0,0] = (x_pri[0]*mcfi_e)**2
b[1, 1] = (x_pri[1]*ch4i_e)**2
b[2, 2] = (x_pri[2]*r13i_e)**2
for i in range(3, nt+3):
    b[i,i] = foh_e**2
    for j in range(3,i):
        b[i,j] = np.exp(-(i-j)/corlen_oh)*(foh_e)**2
        b[j,i] = b[i,j]
for i in range(nt+3, 2*nt+3):
    b[i,i] = fst_e**2
for i in range(2*nt+3, 3*nt+3):
    b[i,i] = fsl_e**2
for i in range(3*nt+3, 4*nt+3):
    b[i,i] = fme_e**2
for i in range(4*nt+3,5*nt+3):
    b[i,i] = fch4_e**2
    for j in range(4*nt+3,i):
        b[i,j] = np.exp(-(i-j)/corlen_em)*fch4_e**2
        b[j,i] = b[i,j]
for i in range(5*nt+3,6*nt+3): 
    b[i,i] = r13e_e**2
#    for j in range(5*nt+3,i):
#        b[i,j] = np.exp(-(i-j)/corlen_em)*r13e_e**2
#        b[j,i] = b[i,j]
        

b_inv = linalg.inv(b)
# Set up preconditioning
L_precon = sqrt(b)
L_adj   = np.transpose(L_precon)
L_inv   = np.linalg.inv(L_precon)
xp_pri = state_to_precon(x_pri)
# The optimization
start = time.time()
xp_opt = optimize.fmin_bfgs(calculate_J,xp_pri,calculate_dJdx, gtol=1e-5)
x_opt = precon_to_state(xp_opt)
end   = time.time()
print 'Optimization run time:',10**3*(end-start),'ms'

mcf_pri = forward_mcf(x_pri)
ch4_pri,r13_pri,c13_pri = forward_ch4(x_pri)
d13c_pri = r13_to_d13c(r13_pri)
_,_,J_ch4_pri  = cost_ch4(ch4_pri)
_,_,J_r13_pri = cost_r13(r13_pri)
_,_,J_mcf_pri = cost_mcf(mcf_pri)

mcf_opt = forward_mcf(x_opt)
ch4_opt,r13_opt,c13_pri = forward_ch4(x_opt)
d13c_opt = r13_to_d13c(r13_opt)
_,_,J_ch4_opt  = cost_ch4(ch4_opt)
_,_,J_r13_opt = cost_r13(r13_opt)
_,_,J_mcf_opt = cost_mcf(mcf_opt)

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
ax1.errorbar(years,ch4_obs,yerr=ch4_obs_e,fmt = 'o',color = 'green',label=r'CH$_4$, observed' )
ax1.plot( years,ch4_pri,'r-',label=r'CH$_4$, prior')
ax1.plot( years,ch4_opt  ,'g-',label=r'CH$_4$, optimized')
ax1b.plot(years,J_ch4_pri, color = 'red')
ax1b.plot(years,J_ch4_opt, color = 'green')
ax2.errorbar(years,d13c_obs, yerr= d13c_obs_e  ,fmt = 'o',color = 'green',label=r'$\delta^{13}$C, observed')
ax2.plot( years,d13c_pri,'r--',label=r'$\delta^{13}$C, prior')
ax2.plot( years,d13c_opt  ,'g--',label=r'$\delta^{13}$C, optimized')
ax2b.plot(years,J_r13_pri, color = 'red')
ax2b.plot(years,J_r13_opt, color = 'green')
ax1.legend(loc='upper left')
ax2.legend(loc='lower right')
if save_fig:
    plt.savefig('d13C_CH4_concentrations_'+exp_name)

fig_mcf = plt.figure(figsize = figSize)
ax1 = fig_mcf.add_subplot(111)
ax1b = ax1.twinx()
ax1.set_xlabel('Year')
ax1b.set_ylabel('Cost function')
ax1.set_ylabel('mcf concentration (ppb)')
ax1.set_title('mcf concentrations: both from observations\n and modelled forward from the prior.')
ax1.errorbar(years,mcf_obs, yerr = mcf_obs_e,fmt='o',color='gray',label = 'mcf observations')
ax1.plot(years,mcf_pri,'-',color='red', label = 'mcf prior')
ax1.plot(years,mcf_opt  ,'-',color='green', label = 'mcf optimized')
ax1b.plot(years,J_mcf_pri, color = 'red')
ax1b.plot(years,J_mcf_opt, color = 'green')
ax1.legend(loc='best')
if save_fig:
    plt.savefig('MCF_concentrations_'+exp_name)

# Plotting prior state vs optimized state
mcfi_opt, ch4i_opt, r13i_opt, foh_opt, fst_opt, fsl_opt, fme_opt, fch4_opt, r13e_opt = unpack(x_opt)  
emcf_opt = em0_mcf + mcf_shift(fst_opt, fsl_opt, fme_opt)
mcf_dev = emcf_opt/em0_mcf - 1.
oh_pri, oh_opt = foh_pri*oh/1.e6, foh_opt*oh/1.e6
errors_oh = np.array([foh_e]*nt)*oh_pri
emcf_pri, emcf_opt = em0_mcf + mcf_shift(fst_pri,fsl_pri,fme_pri), em0_mcf + mcf_shift(fst_opt,fst_opt,fme_opt)
error_e_mcf = sqrt(fst_e**2 + fsl_e**2)
errors_e_mcf = array([error_e_mcf]*nt)*emcf_pri
ech4_pri, ech4_opt = fch4_pri*em0_ch4, fch4_opt*em0_ch4
ed13c_pri, ed13c_opt = r13_to_d13c(r13e_pri), r13_to_d13c(r13e_opt)

fig = plt.figure(figsize = (10,40))
ax1 = fig.add_subplot(411)
ax1.set_title('Prior and optimized global mean OH concentrations')
ax1.set_ylabel('OH concentration \n'+r'($10^6$ molec cm$^{-3}$)')
ax1.plot( years, oh_pri, 'o-', color = 'red',   label = 'Prior'     )
ax1.plot( years, oh_opt  , 'o-', color = 'green', label = 'Optimized' )
ax1.legend(loc='best')
ax2 = fig.add_subplot(412)
ax2.set_title('Prior and optimized global mean MCF emissions')
ax2.set_ylabel('MCF emissions (Gg/yr)')
ax2.plot( years, emcf_pri/1.e6, 'o-', color = 'red',   label = 'Prior'     )
ax2.plot( years, emcf_opt/1.e6  , 'o-', color = 'green', label = 'Optimized' )
ax2.legend(loc='best')
ax3 = fig.add_subplot(413)
ax3.set_title('Prior and optimized '+r'global mean CH$_4$ emissions')
ax3.set_ylabel('CH$_4$ emissions (Tg/yr)')
ax3.plot( years, ech4_pri/1.e9, 'o-', color = 'red',   label = 'Prior'     )
ax3.plot( years, ech4_opt/1.e9  , 'o-', color = 'green', label = 'Optimized' )
ax3.legend(loc='best')
ax4 = fig.add_subplot(414)
ax4.set_title(r'Prior and optimized $\delta^{13}$C global mean CH$_4$ emissions')
ax4.set_xlabel('Year')
ax4.set_ylabel('$\delta^{13}$C (permil)')
ax4.plot( years, ed13c_pri, 'o-', color = 'red',   label = 'Prior'     )
ax4.plot( years, ed13c_opt  , 'o-', color = 'green', label = 'Optimized' )
ax4.legend(loc='best')
if save_fig:
    plt.savefig('full_state_pri_opt_'+exp_name)

fig = plt.figure(figsize = (10,40))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
ax1.set_ylabel('Optimized MCF emi shift (Gg/yr)')
ax2.set_ylabel('fslow deviation (%)')
ax3.set_ylabel('fstock deviation (%)')
ax3.set_xlabel('Year')
ax1.plot( years, mcf_shift(fst_opt, fsl_opt, fme_opt)/1e6, 'o-', color='red' )
ax2.plot( years, 100*(fsl_opt-fsl_pri)  , 'o-', color='green')
ax3.plot( years, 100*(fst_opt-fst_pri), 'o-', color='blue')
ax4.plot( years, 100*(fme_opt-fme_pri), 'o-', color='c')
if save_fig:
    plt.savefig('MCF_emissions_'+exp_name)

rel_dev = (x_opt - x_pri)
fig = plt.figure(figsize = figSize)
ax1 = fig.add_subplot(111)
ax1.set_title('Relative deviations from prior')
ax1.set_xlabel('Year')
ax1.set_ylabel('Relative deviation (%)')
ax1.plot( years, 100.*(foh_opt - foh_pri), 'o-', color = 'blue', label = 'OH')
ax1.plot( years, 100.*mcf_dev, 'o-', color = 'green', label = 'MCF' )
ax1.plot( years, 100.*(fch4_opt - fch4_pri), 'o-', color = 'red', label = r'CH$_4$' )
ax1.legend(loc='best')
if save_fig:
    plt.savefig('rel_dev_from_prior_'+exp_name)

_,_,J_pri = cost_bg(x_opt) # prior
_, _, _, J_pri_foh, J_pri_fst, J_pri_fsl, J_pri_me, J_pri_fch4, J_pri_r13e = unpack(J_pri)

fig_J = plt.figure(figsize = figSize)
ax1 = fig_J.add_subplot(212)
ax2 = fig_J.add_subplot(211)
ax1.set_title('Background part cost function')
ax2.set_title('Observation part cost function')
ax2.set_xlabel('Year')
ax1.set_ylabel('Cost function')
ax2.set_ylabel('Cost function')
ax1.plot( years, J_pri_foh, 'o-', color = 'blue', label = 'OH' )
ax1.plot( years, J_pri_fst+J_pri_fsl, 'o-', color = 'green', label = 'MCF emi' )
ax1.plot( years, J_pri_fch4, 'o-', color = 'red', label = r'CH$_4$ emi' )
ax1.plot( years, J_pri_r13e, 'o-', color = 'maroon', label = r'$\delta^{13}$C of emi' )
ax2.plot( years, J_mcf_opt, 'o-', color = 'green', label = 'MCF' )
ax2.plot( years, J_ch4_opt, 'o-', color = 'red', label = r'CH$_4$' )
ax2.plot( years, J_r13_opt, 'o-', color = 'maroon', label = r'$\delta^{13}$C' )
ax1.legend(loc='best')
ax2.legend(loc='best')
if save_fig:
    plt.savefig('cost_functions_state_'+exp_name)

if write_data:
    # Writing the experiment data to a file in 2 files: One contains the experiment
    # settings, the other the optimized results
    header_p2 = '# Year\tMCF_opt(ppt)\tCH4_opt(ppb)\td13C_opt(perm)\tfsl_opt\t\
                fst_opt\tfch4_opt\ted13C_opt(perm)'
    head_res = header_p1+header_p2
    head_set = '# The settings used in the experiment'
    years = range(styear, edyear)
    write_results(exp_name+'_results.txt',head_res,years,mcf_opt,ch4_opt,d13c_opt,\
                  fsl_opt,fst_opt,fme_opt,fch4_opt,ed13c_opt)
    write_settings(exp_name+'_settings.txt',head_set,b,corlen_oh,corlen_em,years,mcf_obs_e,ch4_obs_e,d13c_obs_e)
    # Save the results as a numpy array
    full_array = np.transpose((years,mcf_opt,ch4_opt,d13c_opt,foh_opt,fst_opt,fsl_opt,fme_opt,fch4_opt,ed13c_opt))
    full_file = os.path.join(os.getcwd(), 'Data output','arrays',exp_name+'_full')
    xopt_file = os.path.join(os.getcwd(), 'Data output','arrays',exp_name+'_xopt')
    np.savetxt(full_file,full_array)
    np.savetxt(xopt_file,x_opt)



fig = plt.figure(figsize=(10,30))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax1.set_title('The number of SD difference between obs\n and optimalized concentrations\n\nMCF')
ax2.set_title(r'CH$_4$')
ax3.set_title(r'$\delta^{13}$C')
ax1.plot(years, (mcf_obs-mcf_opt)/mcf_obs_e, 'go-')
ax1.plot(years, [-1]*nt, 'k--'); ax1.plot(years, [1]*nt, 'k--')
ax2.plot(years, (ch4_obs-ch4_opt)/ch4_obs_e, 'go-')
ax2.plot(years, [-1]*nt, 'k--'); ax2.plot(years, [1]*nt, 'k--')
ax3.plot(years, (d13c_obs-d13c_opt)/d13c_obs_e, 'go-')
ax3.plot(years, [-1]*nt, 'k--'); ax3.plot(years, [1]*nt, 'k--')






mcfff = mcf_obs # For use in data_plots
chhh4 = ch4_obs # For use in data_plots










