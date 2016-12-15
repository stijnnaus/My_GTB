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

red = 1e-2
def calculate_J(xp):
    x = precon_to_state(xp)
    mcf,c12,c13 = forward_all(x)
    ch4,d13c = obs_oper_del(c12,c13)
    dep_mcf, J_mcf, J_list_mcf = cost_mcf(mcf)
    dep_ch4, J_ch4, J_list_ch4 = cost_ch4(ch4)
    dep_d13c, J_d13c,J_list_d13c = cost_d13c(d13c)
    
    J_obs = J_mcf + J_ch4 + J_d13c # mismatch with obs
    print 'Cost mcf:', J_mcf*red/2., 'Cost ch4:', J_ch4*red/2., 'Cost d13c:', J_d13c*red/2.
    
    J_pri = sum(background(x)) # mismatch with prior
    J_tot = .5 * ( J_pri + J_obs )
    print 'Cost observations  :',J_obs*red
    print 'Cost background    :',J_pri*red
    print 'Cost function value:',J_tot*red
    return J_tot*red

def background(x):
    return dot(b_inv, ( x - x_prior )**2)
    
def calculate_dJdx(xp):
    x = precon_to_state(xp)
    _, _, _, foh, _, _, _, _ = unpack(x)
    mcf_save,c12_save,c13_save = forward_all(x)
    ch4, d13c = obs_oper_del(c12_save, c13_save)
    dep_mcf, J_mcf, J_list_mcf = cost_mcf(mcf_save)
    dep_ch4, J_ch4, J_list_ch4 = cost_ch4(ch4)
    dep_d13c, J_d13c, J_list_d13c = cost_d13c(d13c)
    
    dif_ch4, dif_d13c = dep_ch4*ch4_obs_e**2, dep_d13c*d13c_obs_e**2
    dif_c12, dif_c13 = adj_oper_del(dif_ch4, dif_d13c)
    dep_c12, dep_c13 = dif_c12/c12_obs_e**2, dif_c13/c13_obs_e**2
    
    dmcf0,dfoh_mcf,dfst,dfsl = adjoint_model_mcf( dep_mcf, foh, mcf_save )
    dc120,dfoh_c12,dfc12 = adjoint_model_c12( dep_c12, foh, c12_save )
    dc130,dfoh_c13,dfc13 = adjoint_model_c13( dep_c13, foh, c13_save )
    dfoh = dfoh_mcf + dfoh_c12 + dfoh_c13
    
    dJdx_obs = np.concatenate((dmcf0, dc120, dc130, dfoh, \
                                dfst, dfsl, dfc12, dfc13))
    dJdx_pri = dot(b_inv, (x-x_prior))
    dJdx = dJdx_obs + dJdx_pri
    dJdxp = dot( L_adj, dJdx )
    print 'Cost function deriv:',max(dJdxp)*red
    return dJdxp*red

def adjoint_model_mcf( dep, foh, mcf_save ):
    pulse_mcf = adj_oper_av( dep )
    admcf,adfst,adfsl,adfoh = zeros(nt),zeros(nt),zeros(nt),zeros(nt)
    adshift = zeros(nt)
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
    adj_mcf = [array([admcfi]),adfoh,adfst,adfsl]
    return adj_mcf
    
def adjoint_model_c12( dep, foh, c12_save ):
    pulse_c12 = adj_oper_av( dep )
    em0_12 = em0_c12 / conv_ch4
    em0f = em0_12 * dt
    
    df12,dfoh = zeros(nt),zeros(nt)
    dc12i = 0.
    
    for iyear in range(edyear-1, styear-1, -1):
        i = iyear-styear
        for n in range(nstep):
            dc12i += pulse_c12[i][n]
            # Chemistry
            dfoh[i] -= l_ch4_ohf * c12_save[i] * dc12i
            dc12i = dc12i * (1. - foh[i] * l_ch4_ohf - l_ch4_otherf)
            # Emissions
            df12[i] += em0f[i] * dc12i
        
    adj_c12 = [array([dc12i]), dfoh, df12]
    return adj_c12
    
def adjoint_model_c13( dep, foh, c13_save ):
    pulse_c13 = adj_oper_av( dep )
    em0_13 = em0_c13 / conv_c13
    em0f = em0_13*dt
    
    df13,dfoh = zeros(nt),zeros(nt)
    dc13i = 0.
    
    for iyear in range(edyear-1, styear-1, -1):
        i = iyear - styear
        for n in range(nstep):
            dc13i += pulse_c13[i][n]
            # Chemistry
            dfoh[i] -= a_ch4_oh * l_ch4_ohf * c13_save[i] * dc13i
            dc13i   = dc13i * (1 - foh[i] * l_ch4_ohf * a_ch4_oh - a_ch4_other * l_ch4_otherf)
            # Emissions
            df13[i] += em0f[i] * dc13i
        
    adj_c13 = [array([dc13i]), dfoh, df13]
    return adj_c13

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
    
def cost_d13c(d13c):
    dif = (d13c - d13c_obs)
    dep = dif / d13c_obs_e**2
    cost_list = dif*dep
    cost = sum(cost_list)
    return dep,cost,cost_list

def forward_tl_mcf(x, foh, mcf_save):
    dmcf0, _, _, dfoh, dfst, dfsl, _, _ = unpack(x)
    dmcfs = zeros((nt,nstep))
    dmcfi = dmcf0
    rapidc = rapid/conv_mcf
    for year in range(styear,edyear):
        i  = year - styear
        ie = year - 1951
        # Calculate emission shift from fstock ...
        dshift = - 0.75 * rapidc[ie] * dfst[i]
        if i >= 1: dshift -= 0.25 * rapidc[ie-1] * dfst[i-1]
        for yearb in range(year-1,year-11,-1):
            i2  = yearb - styear
            ie2 = yearb - 1951
            if yearb >= styear: dshift += 0.1*dfst[i2]*rapidc[ie2]
        # ... and from fslow
        dshift -= 0.75 * rapidc[ie] * dfsl[i]
        if i >= 2: dshift += 0.75 * rapidc[ie-2] * dfsl[i-2]
        dshiftf = dshift*dt
        # Emission-chemistry interaction
        dmcfis = zeros(nstep)
        for n in range(nstep):
            # Add emission
            dmcfi += dshiftf
            # Chemistry
            dmcfi = dmcfi * ( 1. - l_mcf_oceanf - l_mcf_stratf - l_mcf_ohf * foh[i] ) - \
                    dfoh[i] * mcf_save[i] * l_mcf_ohf
            dmcfs[i][n] = dmcfi
    return obs_oper_av(dmcfs)
    
def forward_tl_c12(x, foh, c12_save):
    _, dc120, _, dfoh, _, _, dfc12, _ = unpack(x)
    dem_c12 = dfc12 * (em0_c12 / conv_ch4)
    demf = dem_c12*dt
    dc12s = zeros((nt,nstep))
    dc12 = dc120
    for year in range(styear,edyear):
        i = year - styear
        for n in range(nstep):
            dc12 += demf[i]
            dc12  = dc12 * ( 1 - l_ch4_otherf - l_ch4_ohf * foh[i]) - \
                    dfoh[i] * c12_save[i] * l_ch4_ohf
            dc12s[i][n] = dc12
    return obs_oper_av(dc12s)
    
def forward_tl_c13(x, foh, c13_save):
    _, _, dc130, dfoh, _, _, _, dfc13 = unpack(x)
    dem_c13 = dfc13 * (em0_c13 / conv_c13)
    demf = dem_c13*dt
    dc13s = zeros((nt,nstep))
    dc13 = dc130
    for year in range(styear,edyear):
        i = year - styear
        for n in range(nstep):
            dc13 += demf[i]
            dc13  = dc13 * ( 1 - a_ch4_other * l_ch4_otherf - a_ch4_oh * l_ch4_ohf * foh[i] ) - \
                    dfoh[i] * c13_save[i] * l_ch4_ohf * a_ch4_oh
            dc13s[i][n] = dc13
    return obs_oper_av(dc13s)

def forward_all(x):
    C_mcf,C_c12,C_c13 = forward_mcf(x),forward_c12(x),forward_c13(x)
    return C_mcf,C_c12,C_c13

def forward_mcf(x):
    mcf0, _, _, foh, fst, fsl, _, _ = unpack(x)
    em = em0_mcf + mcf_shift(fst, fsl)
    em /= conv_mcf
    mcfs = zeros((nt,nstep))
    mcf = mcf0
    for year in range(styear,edyear):
        i = year - styear
        emf = em[i]*dt
        for n in range(nstep):
            mcf += emf
            mcf = mcf * ( 1 - l_mcf_ohf * foh[i] - l_mcf_oceanf - l_mcf_stratf )
            mcfs[i][n] = mcf
    return obs_oper_av(mcfs)
    
def forward_c12(x):
    _, c120, _, foh, _, _, fc12, _ = unpack(x)
    em = fc12 * (em0_c12 / conv_ch4)
    
    c12s = zeros((nt,nstep))
    c12 = c120
    for year in range(styear,edyear):
        i = year - styear
        emf = em[i]*dt
        for n in range(nstep):
            c12 += emf
            c12  = c12 * ( 1 - l_ch4_otherf - l_ch4_ohf * foh[i])
            c12s[i][n] = c12
    return obs_oper_av(c12s)
   
def forward_c13(x):
    _, _, c130, foh, _, _, _, fc13 = unpack(x)
    em = fc13 * (em0_c13 / conv_c13)
    c13s = zeros((nt,nstep))
    c13 = c130
    for year in range(styear,edyear):
        i = year - styear
        emf = em[i]*dt
        for n in range(nstep):
            c13 += emf
            c13  = c13 * ( 1 - a_ch4_other * l_ch4_otherf - a_ch4_oh * l_ch4_ohf * foh[i] )
            c13s[i][n] = c13
    return obs_oper_av(c13s)

def mcf_shift(fst, fsl):
    '''
    Computes the shift in emissions that results from repartitioning of the production
    fst: the fraction of rapid emissions that is instead stockpiled
    fsl: the fraction of rapid emissions that is instead released as slow
    ''' 
    shifts = []
    for year in range(styear,edyear):
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
        shifts.append(shift)
    
    return array(shifts)
    
def obs_oper_del(c12, c13):
    '''Converts 12CH4 and 13CH4 concentrations to CH4 concentration and d13C'''
    ch4 = c12 + c13
    d13c = 1000 * (( (c13/c12) / R_ST ) - 1)
    return ch4, d13c
    
def adj_oper_del(ch4, d13c):
    '''Converts CH4 concentration and d13C to 12CH4 and 13CH4 concentrations'''
    q = array( R_ST * (d13c/1000.+1.) )
    c12 = ch4 / (q+1)
    c13 = q*c12
    return c12, c13
    
def obs_oper_av(con):
    '''Converts from nsteps per year to yearly averages. '''
    means = zeros(nt)
    for i,c in enumerate(con):
        means[i] = sum(c) / nstep
    return means
    
def adj_oper_av(means):
    '''Converts a yearly average to pulses per nstep.'''
    pulses = []
    for mean in means:
        pulse = [mean for i in range(nstep)]
        pulses.append(array(pulse))
    pulses = array(pulses)/nstep
    return pulses
    
def state_to_precon(x):
    ''' Convert the state and derivative to the preconditioned space. '''
    #return x
    return dot( L_inv, (x - x_prior) ) 

def precon_to_state(xp):
    ''' Convert the preconditioned state to the original space. '''
    #return xp
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
    ''' Unpacks the state vector. Returns the respective components. '''
    mcf0,c120,c130 = x[0],x[1],x[2]
    foh = x[3:nt+3]
    fstock,fslow = x[nt+3:2*nt+3],x[2*nt+3:3*nt+3]
    fc12,fc13 = x[3*nt+3:4*nt+3],x[4*nt+3:5*nt+3]
    
    return mcf0, c120, c130, foh, fstock, fslow, fc12, fc13    

# setup model variables, emissions, etc:
exp_name = '_Timeres400'
nstep = 400
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
oh = .9*1e6  # molecules/cm3
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
# Calculate loss at the increased timeres frequency
l_ch4_otherf = l_ch4_other * dt
l_ch4_ohf = l_ch4_oh * dt
l_mcf_ohf = l_mcf_oh * dt
l_mcf_oceanf = l_mcf_ocean * dt
l_mcf_stratf = l_mcf_strat * dt
styear,edyear = 1988,2009

nt = edyear-styear

# Reading in the data
mcf_obs,mcf_obs_e = read_mcf_measurements()
rapid,medium,slow,stock,em0_mcf,prod = read_mcf_emi(os.path.join('EMISSIONS','emissions.dat'))
ch4_obs,ch4_obs_e = read_ch4_measurements()
em0_ch4 = array([550.0]*nt)*1e9
d13c_obs,d13c_obs_e = read_d13C_obs(os.path.join('OBSERVATIONS','d13C_Schaefer.txt'))
em0_d13c = array([-54.1]*nt)

fec,femcf = 1.5,1.5 # Reduction of the error
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
foh_prior = ones(nt)
fsl_prior = zeros(nt)
fst_prior = zeros(nt)
f12_prior = ones(nt)
f13_prior = ones(nt)

x_prior = concatenate((mcf0_prior, c120_prior, c130_prior, foh_prior, \
                       fst_prior,  fsl_prior,  f12_prior,  f13_prior))

nstate = len(x_prior)
# Constructing the prior error matrix b
b = zeros((nstate,nstate))
error_oh = .05
error_e_st = .02; error_e_sl = .02   # mcf emission errors
error_e_c12 = .05; error_e_c13 = .05 # ch4 emission errors
error_mcf0 = .2; error_c120 = .2; error_c130 = .2 # error in initial concentration
corlen_oh = 1. 
corlen_em = 1.

b[0,0] = (x_prior[0]*error_mcf0)**2
b[1, 1] = (x_prior[1]*error_c120)**2
b[2, 2] = (x_prior[2]*error_c130)**2
for i in range(3, nt+3):
    b[i,i] = error_oh**2
    for j in range(3,i):
        b[i,j] = exp(-(i-j)/corlen_oh)*(error_oh)**2
        b[j,i] = b[i,j]
for i in range(nt+3, 2*nt+3):
    b[i,i] = error_e_st**2
for i in range(2*nt+3,3*nt+3):
    b[i,i] = error_e_c12**2
    for j in range(3*nt+3,i):
        b[i,j] = exp(-(i-j)/corlen_em)*(error_e_c12)**2
        b[j,i] = b[i,j]
for i in range(4*nt+3,5*nt+3): 
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
xp_opt = optimize.fmin_bfgs(calculate_J,xp_prior,calculate_dJdx, gtol=1e-3)
x_opt = precon_to_state(xp_opt)
end   = time.time()
print 'Optimization run time:',10**3*(end-start),'ms'

mcf_prior = forward_mcf(x_prior)
c12_prior,c13_prior = forward_c12(x_prior),forward_c13(x_prior)
ch4_prior,d13c_prior = obs_oper_del(c12_prior, c13_prior)
_,_,J_ch4_prior  = cost_ch4(ch4_prior)
_,_,J_d13c_prior = cost_d13c(d13c_prior)
_,_,J_mcf_prior = cost_mcf(mcf_prior)
J_c12_prior = ((c12_prior-c12_obs) / c12_obs_e)**2
J_c13_prior = ((c12_prior-c12_obs) / c12_obs_e)**2

mcf_opt = forward_mcf(x_opt)
c12_opt, c13_opt = forward_c12(x_opt), forward_c13(x_opt)
ch4_opt, d13c_opt = obs_oper_del(c12_opt, c13_opt)
_,_,J_ch4_opt  = cost_ch4(ch4_opt)
_,_,J_d13c_opt = cost_d13c(d13c_opt)
_,_,J_mcf_opt = cost_mcf(mcf_opt)
J_c12_opt = ((c12_opt-c12_obs) / c12_obs_e)**2
J_c13_opt = ((c13_opt-c13_obs) / c13_obs_e)**2

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

mcf0_opt, c120_opt, c130_opt, foh_opt, fst_opt, fsl_opt, f12_opt, f13_opt = unpack(x_opt)  
em0_opt = em0_mcf + mcf_shift( fst_opt, fsl_opt )
mcf_dev = em0_opt/em0_mcf - 1.


oh_prior, oh_opt = foh_prior*oh/1.e6, foh_opt*oh/1.e6
errors_oh = np.array([error_oh]*nt)*oh_prior
emcf_prior, emcf_opt = em0_mcf + mcf_shift(fst_prior,fsl_prior), em0_mcf + mcf_shift(fst_opt,fst_opt)
error_e_mcf = sqrt(error_e_st**2 + error_e_sl**2)
errors_e_mcf = array([error_e_mcf]*nt)*emcf_prior
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
ax1.plot( range(styear,edyear), 100.*(foh_opt - foh_prior), 'o-', color = 'blue', label = 'OH')
ax1.plot( range(styear,edyear), 100.*mcf_dev, 'o-', color = 'green', label = 'MCF' )
ax1.plot( range(styear,edyear), 100.*(f12_opt - f12_prior), 'o-', color = 'red', label = r'$^{12}$CH$_4$' )
ax1.plot( range(styear,edyear), 100.*(f13_opt - f13_prior), 'o-', color = 'maroon', label = r'$^{13}$CH$_4$' )
ax1.legend(loc='best')
plt.savefig('rel_dev_from_prior'+exp_name)

J_prior = background(x_opt) # prior
_, _, _, J_pri_foh, J_pri_fst, J_pri_fsl, J_pri_f12, J_pri_f13 = unpack(J_prior)
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
ax1.plot( range(styear,edyear), J_pri_foh, 'o-', color = 'blue', label = 'OH' )
ax1.plot( range(styear,edyear), J_pri_fst+J_pri_fsl, 'o-', color = 'green', label = 'MCF' )
ax1.plot( range(styear,edyear), J_pri_f12, 'o-', color = 'red', label = r'$^{12}$CH$_4$' )
ax1.plot( range(styear,edyear), J_pri_f13, 'o-', color = 'maroon', label = r'$^{13}$CH$_4$' )
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






delc = np.linspace(-5000.,5000.)/1000.
ch4 = 1500.
q = array( R_ST * (delc+1) )
c12 = ch4 / (q+1)
c13 = q*c12

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.plot(delc,c12, 'ro')
ax2.plot(delc,c13, 'bo')














