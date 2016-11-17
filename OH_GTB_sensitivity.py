# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 11:36:16 2016

@author: Stijn
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

def calculate_J(xp):
    x = precon_to_state(xp)
    con = forward_all(x)
    dep_mcf,J_mcf   = cost_mcf(con)
    dep_ch4,J_ch4   = cost_ch4(con)
    dep_d13c,J_d13c = cost_d13c(con)
    
    J_pri = sum(dot(b_inv, ( x - x_prior )**2)) # prior
    J_obs = J_mcf + J_ch4 + J_d13c # mismatch with obs
    J_tot = J_pri + J_obs
    print 'Cost function value:',J_tot
    return J_tot
    
def calculate_dJdx(xp):
    x = precon_to_state(xp)
    foh = x[:nt]
    mcf_save,c12_save,c13_save = forward_all(x)
    
    con = forward_all(x)
    dep_mcf,J_mcf = cost_mcf(con)
    dep_ch4,J_ch4 = cost_ch4(con)
    dep_d13c,J_d13c = cost_d13c(con)
    dep = array([dep_mcf,dep_ch4,dep_d13c])
    
    dfoh_mcf,dmcfi,dfmcf = adjoint_model_mcf( dep, foh, mcf_save )
    dfoh_c12,dc12i,df12 = adjoint_model_c12( dep, foh, c12_save )
    dfoh_c13,dc13i,df13 = adjoint_model_c13( dep, foh, c13_save )
    dfoh = dfoh_mcf + dfoh_c12 + dfoh_c13
    
    dJdx_obs = np.concatenate((dfoh, dmcfi, dfmcf, \
                                dc12i, df12, dc13i, df13))
    dJdx_pri = dot(b_inv, (x-x_prior))
    dJdx = dJdx_obs + dJdx_pri
    dJdxp = dot( L_adj, dJdx )
    print 'Cost function deriv:',max(dJdxp)
    return dJdxp

def adjoint_model_mcf( dep, foh, mcf_save ):
    pulse_mcf = adj_obs_oper( dep )[0]
    dmcf = zeros(nt)
    dfmcf,dfoh = zeros(nt),zeros(nt)
    dmcfi = 0.
    rapidc = rapid/conv_mcf
    
    for i in range(edyear-1,styear-1,-1):
        iyear = i-styear
        eyear = i-1951
        
        # Add adjoint pulses
        dmcfi   += pulse_mcf[iyear]
        
        # Chemistry
        dfoh[iyear] = - l_mcf_oh * mcf_save[iyear] * dmcfi
        dmcfi       = dmcfi *  (1. - foh[iyear] * l_mcf_oh - l_mcf_strat - l_mcf_ocean)
        dmcf[iyear] = dmcfi
        
        # Emissions
        dstock = 0.
        for j in range(i+1 , i+11):
            jyear = j - styear
            if j < edyear: dstock += 0.1 * rapidc[eyear] * dmcf[jyear]
        dfmcf[iyear] = dfmcf[iyear] - 0.75 * rapidc[eyear] * dmcf[iyear]  + dstock
        if (i + 1) < edyear: dfmcf[iyear] -= -0.25 * rapidc[eyear] * dmcf[iyear+1]
            
    adj_mcf = [dfoh, array([dmcfi]), dfmcf]
    return adj_mcf
    
def adjoint_model_c12( dep, foh, c12_save ):
    pulse_c12 = adj_obs_oper( dep )[1]
    em0_12 = em0_c12 / conv_ch4
    
    df12,dfoh = zeros(nt),zeros(nt)
    dc12i = 0
    
    for i in range(edyear-1, styear-1, -1):
        iyear = i-styear
        dc12i += pulse_c12[iyear]
        
        dfoh[iyear] = - l_ch4_oh * c12_save[iyear] * dc12i
        dc12i = dc12i * (1 - foh[iyear] * l_ch4_oh - l_ch4_other)
        
        df12[iyear] = em0_12[iyear] * dc12i
        
    adj_c12 = [dfoh, array([dc12i]), df12]
    return adj_c12
    
def adjoint_model_c13( dep, foh, c13_save ):
    pulse_c13 = adj_obs_oper( dep )[2]
    em0_13 = em0_c13 / conv_c13
    
    df13,dfoh = zeros(nt),zeros(nt)
    dc13i = 0
    
    for i in range(edyear-1, styear-1, -1):
        iyear = i-styear
        dc13i += pulse_c13[iyear]
        
        dfoh[iyear] = - l_ch4_oh * c13_save[iyear] * dc13i * a_ch4_oh
        dc13i = dc13i * (1 - foh[iyear] * l_ch4_oh * a_ch4_oh - a_ch4_other * l_ch4_other)
        
        df13[iyear] = em0_13[iyear] * dc13i
        
    adj_c13 = [dfoh, array([dc13i]), df13]
    return adj_c13

def cost_mcf(con):
    mcf,ch4,d13c = obs_oper(con)
    dif = (mcf - mcf_obs)
    dep = dif / mcf_obs_e**2
    cost = sum(dif*dep)
    return dep,cost
    
def cost_ch4(con):
    mcf,ch4,d13c = obs_oper(con)
    dif = (ch4 - ch4_obs)
    dep = dif / ch4_obs_e**2
    cost = sum(dif*dep)
    return dep,cost
    
def cost_d13c(con):
    mcf,ch4,d13c = obs_oper(con)
    dif = (d13c - d13c_obs)
    dep = dif / d13c_obs_e**2
    cost = sum(dif*dep)
    return dep,cost

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
    dem0_12 = dfc12 * (em0_c12 / conv_ch4)
    
    dc12s = []; dc12 = dc12_0
    for year in range(styear,edyear):
        i = year - styear
        dc12 += dfc12[i] * dem0_12[i]
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
    mcf = con[0]
    c12 = con[1]
    c13 = con[2]
    ch4,d13c = split_to_deltot(c12,c13)
    obs = array([mcf,ch4,d13c])
    return obs

def adj_obs_oper(dep):
    mcf = dep[0]
    ch4 = dep[1]
    d13c = dep[2]
    c12,c13 = deltot_to_split(ch4,d13c)
    adj = array([mcf,c12,c13])
    return adj
    
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
oh0 = 0.70e6  # molecules/cm3
oh = oh0
temp = 272.0  # Kelvin        #
l_mcf_oh0 = (1.64e-12*exp(-1520.0/temp))*oh  # in s-1
l_ch4_oh0 = (2.45e-12*exp(-1775.0/temp))*oh  
l_mcf_oh0 *= 3600.*24.0*365.0  # in yr-1
l_ch4_oh0 *= 3600.*24.0*365.0
l_mcf_oh = l_mcf_oh0
l_ch4_oh = l_ch4_oh0
a_ch4_oh = 1 - 3.9/1000 # fractionation by OH
R_ST = 11237.2e-6 # Standard V-PDP 13C/12C ratio
l_ch4_oh_abs = 528. # Tg yr-1
l_ch4_other_abs = 109. # Tg yr-1
t_ch4 = 9.1 # lifetime methane in yr
l_ch4_other0 = l_ch4_other_abs / (t_ch4 * (l_ch4_oh_abs +l_ch4_other_abs)) # in yr-1
l_ch4_other  = l_ch4_other0
a_ch4_other0 = 1 - 19./1000
a_ch4_other  = a_ch4_other0
styear,edyear = 1988,2009

nt = edyear-styear

# Reading in the data
mcf_obs,mcf_obs_e = read_mcf_measurements()
rapid,medium,slow,stock,em0_mcf,prod = read_mcf_emi(os.path.join('EMISSIONS','emissions.dat'))
ch4_obs,ch4_obs_e = read_ch4_measurements()
em0_ch4 = array([550.0]*nt)*1e9
d13c_obs,d13c_obs_e = read_d13C_obs(os.path.join('OBSERVATIONS','d13C_Schaefer.txt'))
em0_d13c0 = array([-54.1]*nt)
em0_d13c = em0_d13c0
c12_obs, c13_obs, c12_obs_e, c13_obs_e = deltot_to_split(ch4_obs,d13c_obs,ch4_obs_e,d13c_obs_e)
em0_c120, em0_c130 = deltot_to_split(em0_ch4, em0_d13c,mass=True)
em0_c12, em0_c13 = em0_c120, em0_c130
con_data = array([mcf_obs,ch4_obs,d13c_obs])
con_data_e = array([mcf_obs_e,ch4_obs_e,d13c_obs_e])

mcf_ini0 = array([117.])
mcf_ini  = mcf_ini0
ch4_ini0 = ch4_obs[0]/1.01
ch4_ini  = ch4_ini0
d13c_ini0 = d13c_obs[0]
d13c_ini  = d13c_ini0
c12_ini,c13_ini = deltot_to_split(ch4_ini,d13c_ini)
c12_ini,c13_ini = array([c12_ini]),array([c13_ini])
foh_prior  = ones(nt)
fmcf_prior = zeros(nt)
f12_prior  = ones(nt)
f13_prior  = ones(nt)

x_prior = concatenate((foh_prior, mcf_ini, fmcf_prior, \
                        c12_ini, f12_prior, c13_ini, f13_prior))


sens_test = array([0.80,0.90,0.95,0.98,1.,1.02,1.05,1.1,1.2])  # molecules/cm3
colori = array(['darkblue','red','grey','maroon','black','maroon','grey','red','darkblue'])
markeri = array(['-']*9)
obs_color = 'darkgreen'
c12_prior,c13_prior = forward_c12(x_prior),forward_c13(x_prior)
ch4_prior,d13c_prior = split_to_deltot(c12_prior,c13_prior)
mcf_prior = forward_mcf(x_prior)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                       Sensitivity to OH
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print 'Checking sensitivity to OH scaling ... '

ch4_oh,d13c_oh,mcf_oh = [],[],[]
for sens in sens_test:
    l_mcf_oh = sens * l_mcf_oh0
    l_ch4_oh = sens * l_ch4_oh0
    c12_priori,c13_priori = forward_c12(x_prior),forward_c13(x_prior)
    ch4_priori,d13c_priori = split_to_deltot(c12_priori,c13_priori)
    mcf_priori = forward_mcf(x_prior)
    
    ch4_oh.append(ch4_priori)
    d13c_oh.append(d13c_priori)
    mcf_oh.append(mcf_priori)
    
f, axarr = plt.subplots(3, sharex = True,figsize=(15,15))
ax1,ax2,ax3 = axarr[0],axarr[1],axarr[2]
ax1.set_title('OH sensitivity test\n \
               OH change relative to reference is given in the legend.\n \
               Reference value: ' + r'$0.70 \cdot 10^{6}$ molec cm$^{-3}$')
ax1.set_ylabel(r'CH$_4$ concentration (ppb)')
ax2.set_ylabel(r'$\delta^{13}$C (permil)')
ax3.set_ylabel(r'MCF concentration (ppb)')
ax3.set_xlabel('Year')
ax1.plot(range(styear,edyear),ch4_obs,'o',color = obs_color,label= 'Observed' )
ax2.plot(range(styear,edyear),d13c_obs,'o',color = obs_color,label= 'Observed')
ax3.plot(range(styear,edyear),mcf_obs,'o',color = obs_color, label = 'Observed')
for i in range(len(sens_test)-1,-1,-1):
    if sens_test[i]-1 == 0:
        labeli = str(100.*(sens_test[i]-1.)) + '%'
    elif sens_test[i]-1 > 0.:
        labeli = '+ '+str(100.*(sens_test[i]-1.)) + '%'
    else:
        labeli = '- '+str(100.*(1.-sens_test[i])) + '%'
    ax1.plot(range(styear,edyear),ch4_oh[i],markeri[i],color = colori[i],label=labeli)
    ax2.plot(range(styear,edyear),d13c_oh[i],markeri[i],color = colori[i],label=labeli)
    ax3.plot(range(styear,edyear),mcf_oh[i],markeri[i],color = colori[i],label=labeli)
lgd = ax2.legend(bbox_to_anchor = (1.2, 1.07))
plt.savefig('Sensitivity_test_OH',bbox_extra_artists=(lgd,), bbox_inches='tight')

l_mcf_oh = l_mcf_oh0
l_ch4_oh = l_ch4_oh0
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                   Sensitivity to emissions
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print 'Checking sensitivity to absolute emission changes/shifts ... '

sens_test = array([0.94,0.96,0.98,0.99,1.,1.01,1.02,1.04,1.06])
ch4_em, d13c_em, mcf_em = [],[],[]
for sens in sens_test:
    fmcf_priori = zeros(nt) + 7*(sens-1)
    f12_priori  = ones(nt) * sens
    f13_priori  = ones(nt) * sens
    
    x_priori = concatenate((foh_prior, mcf_ini, fmcf_priori, \
                            c12_ini, f12_priori, c13_ini, f13_priori))
    
    c12_priori,c13_priori = forward_c12(x_priori),forward_c13(x_priori)
    ch4_priori,d13c_priori = split_to_deltot(c12_priori,c13_priori)
    mcf_priori = forward_mcf(x_priori)
    
    ch4_em.append(ch4_priori); d13c_em.append(d13c_priori)
    mcf_em.append(mcf_priori)

f, axarr = plt.subplots(3, sharex = True,figsize=(15,15))
ax1,ax2,ax3 = axarr[0],axarr[1],axarr[2]
ax1.set_title('Emission sensitivity test \n \
               For CH4 emission changes are relative to the default (550 Tg yr-1) \n \
               For MCF f_mcf is set at the given values (ie emissions are shifted in time)' )
ax1.set_ylabel(r'CH$_4$ concentration (ppb)')
ax2.set_ylabel(r'$\delta^{13}$C (permil)')
ax3.set_ylabel(r'MCF concentration (ppb)')
ax3.set_xlabel('Year')
ax1.plot(range(styear,edyear),ch4_obs,'o',color = obs_color,label= 'Observed' )
ax2.plot(range(styear,edyear),d13c_obs,'o',color = obs_color,label= 'Observed')
ax3.plot(range(styear,edyear),mcf_obs,'o',color = obs_color, label = 'Observed')
for i in range(len(sens_test)-1,-1,-1):
    if sens_test[i]-1 == 0:
        labeli = str(100.*(sens_test[i]-1.)) 
    elif sens_test[i]-1 > 0.:
        labeli = '+ '+str(100.*(sens_test[i]-1.)) 
    else:
        labeli = '- '+str(100.*(1.-sens_test[i])) 
    ax1.plot(range(styear,edyear),ch4_em[i],markeri[i],color = colori[i],label= labeli + '%')
    ax2.plot(range(styear,edyear),d13c_em[i],markeri[i],color = colori[i],label=labeli + '%')
    ax3.plot(range(styear,edyear),mcf_em[i],markeri[i],color = colori[i],label='f_mcf = ' + str(2*(sens_test[i]-1)))
lgd = ax2.legend(bbox_to_anchor = (1.18, 1.5))
lgd2 = ax3.legend(bbox_to_anchor = (1.2,1.10))
plt.savefig('Sensitivity_test_Emission',bbox_extra_artists=(lgd,lgd2,), bbox_inches='tight')
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#            Sensitivity to d13C of CH4 emissions
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print 'Checking sensitivity to d13C of CH4 emissions ... '

d13c_ch, ch4_ch = [], []
labels = []
num = range(9)
for i in num:
    em0_d13c = em0_d13c0 + (i-4) * 0.05
    labels.append(str(em0_d13c[0]))
    em0_c12,em0_c13 = deltot_to_split(em0_ch4, em0_d13c,mass=True)
    
    c12_priori,c13_priori  = forward_c12(x_prior),forward_c13(x_prior)
    ch4_priori,d13c_priori = split_to_deltot(c12_priori,c13_priori)
    
    ch4_ch.append(ch4_priori)
    d13c_ch.append(d13c_priori)
    
f = plt.figure(figsize = (15,5))
ax = f.add_subplot(111)
ax.set_title(r'$\delta^{13}$C sensitivity test'+'\n \
               The source value is changed to the value given in the legend (default is '+str(em0_d13c0[0])+' permil)')
ax.set_ylabel(r'$\delta^{13}$C (permil)')
ax.set_xlabel('Year')
ax.plot(range(styear,edyear),d13c_obs,'o',color = obs_color,label= 'Observed')
for i in range(8,-1,-1):
    ax.plot(range(styear,edyear),d13c_ch[i],markeri[i],color = colori[i],label=labels[i])
lgd = ax.legend(bbox_to_anchor = (1.2,1.10))
plt.savefig('Sensitivity_test_d13C',bbox_extra_artists=(lgd,), bbox_inches='tight')

em0_d13c = em0_d13c0
em0_c12,em0_c13  = deltot_to_split(em0_ch4, em0_d13c,mass=True)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           Sensitivity to initial concentration values
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print 'Checking sensitivity to initial values ... '

factors = array([0.96,0.98,0.99,0.995,1.0,1.005,1.01,1.02,1.04])
ch4_inch4,d13c_inch4   = [],[]  # Change CH4 initial value
ch4_ind13c,d13c_ind13c = [],[]  # Change d13C initial value
mcf_inmcf = []                  # Change mcf initial value

# ---------------------- CH4 ini shift ------------------------

for fac in factors:
    ch4_ini = ch4_ini0 * fac
    c12_ini,c13_ini = deltot_to_split(ch4_ini,d13c_ini)
    c12_ini,c13_ini = array([c12_ini]),array([c13_ini])
    
    x_priori = concatenate((foh_prior, mcf_ini, fmcf_prior, \
                        c12_ini, f12_prior, c13_ini, f13_prior))
    
    c12_priori,c13_priori  = forward_c12(x_priori),forward_c13(x_priori)
    ch4_priori,d13c_priori = split_to_deltot(c12_priori,c13_priori)
    
    ch4_inch4.append(ch4_priori)
    d13c_inch4.append(d13c_priori)
    
f,axarr = plt.subplots(2,sharex=True,figsize = (15,10))
ax1,ax2 = axarr[0],axarr[1]
ax1.set_title('Initial value of CH4 sensitivity test \n \
               The starting value of CH4 (default '+str(round(ch4_ini0,0))+') was changed by the percentages given in the legend')
ax1.set_ylabel(r'CH$_4$ concentration (ppb)')
ax2.set_ylabel(r'$\delta^{13}$C (permil)')
ax2.set_xlabel('Year')
ax1.plot(range(styear,edyear),ch4_obs,'o',color = obs_color,label= 'Observed' )
ax2.plot(range(styear,edyear),d13c_obs,'o',color = obs_color,label= 'Observed')
for i in range(len(factors)-1,-1,-1):
    if factors[i]-1 == 0:
        labeli = str(100.*(factors[i]-1.)) 
    elif sens_test[i]-1 > 0.:
        labeli = '+ '+str(100.*(factors[i]-1.)) 
    else:
        labeli = '- '+str(100.*(1.-factors[i]))
    ax1.plot(range(styear,edyear),ch4_inch4[i] ,markeri[i],color = colori[i],label=labeli+'%')
    ax2.plot(range(styear,edyear),d13c_inch4[i],markeri[i],color = colori[i],label=labeli+'%')
lgd = ax2.legend(bbox_to_anchor = (1.18, 1.5))
plt.savefig('Sensitivity_test_iniCH4',bbox_extra_artists=(lgd,), bbox_inches='tight')

ch4_ini = ch4_ini0
c12_in,c13_ini = deltot_to_split(ch4_ini,d13c_ini)

# ---------------------- d13C ini shift ------------------------

shifts = array([-0.2,-0.1,-0.05,-0.02,0.,0.02,0.05,0.1,0.2])
labels = []
for shift in shifts:
    d13c_ini = d13c_ini0 + shift
    labels.append(d13c_ini)
    c12_ini,c13_ini = deltot_to_split(ch4_ini,d13c_ini)
    c12_ini,c13_ini = array([c12_ini]),array([c13_ini])
    
    x_priori = concatenate((foh_prior, mcf_ini, fmcf_prior, \
                        c12_ini, f12_prior, c13_ini, f13_prior))
    
    c12_priori,c13_priori  = forward_c12(x_priori),forward_c13(x_priori)
    ch4_priori,d13c_priori = split_to_deltot(c12_priori,c13_priori)
    
    ch4_ind13c.append(ch4_priori)
    d13c_ind13c.append(d13c_priori)
    
f = plt.figure(figsize = (15,5))
ax = f.add_subplot(111)
ax.set_title('Initial value of d13C sensitivity test \n \
               The starting value of d13C (default '+str(d13c_ini0)+') was changed by the values given in the legend')
ax.set_ylabel(r'$\delta^{13}$C (permil)')
ax.set_xlabel('Year')
ax.plot(range(styear,edyear),d13c_obs,'o',color = obs_color,label= 'Observed')
for i in range(len(shifts)-1,-1,-1):
    ax.plot(range(styear,edyear),d13c_ind13c[i],markeri[i],color = colori[i],label=str(shifts[i]))
lgd = ax.legend(bbox_to_anchor = (1.18, 1.05))
plt.savefig('Sensitivity_test_inid13C',bbox_extra_artists=(lgd,), bbox_inches='tight')

d13c_ini = d13c_ini0
c12_ini,c13_ini = deltot_to_split(ch4_ini,d13c_ini)
c12_ini,c13_ini = array([c12_ini]),array([c13_ini])

# ---------------------- MCF ini shift ------------------------

factors = array([0.92,0.95,0.97,0.99,1.0,1.01,1.03,1.05,1.08])
for fac in factors:
    mcf_ini = mcf_ini0*fac
    
    x_priori = concatenate((foh_prior, mcf_ini, fmcf_prior, \
                        c12_ini, f12_prior, c13_ini, f13_prior))
    
    mcf_priori = forward_mcf(x_priori)
    
    mcf_inmcf.append(mcf_priori)
    
f = plt.figure(figsize = (15,5))
ax = f.add_subplot(111)
ax.set_title('Initial value of MCF sensitivity test \n \
               The starting value of MCF (default '+str(mcf_ini0)+') was changed by the percentages given in the legend')
ax.set_ylabel('MCF concentration (ppb)')
ax.set_xlabel('Year')
ax.plot(range(styear,edyear),mcf_obs,'o',color = obs_color,label= 'Observed')
for i in range(len(factors)-1,-1,-1):
    if factors[i]-1 == 0:
        labeli = str(100.*(factors[i]-1.)) 
    elif sens_test[i]-1 > 0.:
        labeli = '+ '+str(100.*(factors[i]-1.)) 
    else:
        labeli = '- '+str(100.*(1.-factors[i]))
    ax.plot(range(styear,edyear),mcf_inmcf[i],markeri[i],color = colori[i],label=labeli + '%')
lgd = ax.legend(bbox_to_anchor = (1.18, 1.05))
plt.savefig('Sensitivity_test_iniMCF',bbox_extra_artists=(lgd,), bbox_inches='tight')

mcf_ini = mcf_ini0

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           Sensitivity to other sink of CH4
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print 'Checking sensitivity to the other sink ...'

# ------------------------- Other magnitude -------------------------

ch4_othabs, d13c_othabs = [],[]
factors = array([0.94,0.96,0.98,0.99,1.,1.01,1.02,1.04,1.06])
for fac in factors:
    l_ch4_other = l_ch4_other0 * fac
    
    c12_priori,c13_priori  = forward_c12(x_priori),forward_c13(x_priori)
    ch4_priori,d13c_priori = split_to_deltot(c12_priori,c13_priori)
    
    ch4_othabs.append(ch4_priori)
    d13c_othabs.append(d13c_priori)
    
f,axarr = plt.subplots( 2, sharex = True, figsize = (15,10) )
ax1,ax2 = axarr[0],axarr[1]
ax1.set_title('Test for sensitivity to the absolute size of the other sink of CH4 \n \
               The other sink size was adjusted by the percentages given in the legend')
ax1.set_ylabel(r'CH$_4$ concentration (ppb)')
ax2.set_ylabel(r'$\delta^{13}$C (permil)')
ax2.set_xlabel('Year')
ax1.plot(range(styear,edyear),ch4_obs,'o',color = obs_color,label= 'Observed' )
ax2.plot(range(styear,edyear),d13c_obs,'o',color = obs_color,label= 'Observed')
for i in range(len(factors)-1,-1,-1):
    if factors[i]-1 == 0:
        labeli = str(100.*(factors[i]-1.)) 
    elif sens_test[i]-1 > 0.:
        labeli = '+ '+str(100.*(factors[i]-1.)) 
    else:
        labeli = '- '+str(100.*(1.-factors[i]))
    ax1.plot(range(styear,edyear),ch4_othabs[i] ,markeri[i],color = colori[i],label=labeli+'%')
    ax2.plot(range(styear,edyear),d13c_othabs[i],markeri[i],color = colori[i],label=labeli+'%')
lgd = ax2.legend(bbox_to_anchor = (1.18, 1.5))
plt.savefig('Sensitivity_test_CH4other_abs',bbox_extra_artists=(lgd,), bbox_inches='tight')

l_ch4_other = l_ch4_other0

# ---------------------- Other fractionation ------------------------

ch4_othfrac, d13c_othfrac = [],[]
epsis = array([17.,17.5,18.,18.5,19.,19.5,20.,20.5,21.])
for epsi in epsis:
    a_ch4_other = 1 - epsi/1000.
    
    c12_priori,c13_priori  = forward_c12(x_priori),forward_c13(x_priori)
    ch4_priori,d13c_priori = split_to_deltot(c12_priori,c13_priori)
    
    ch4_othfrac.append(ch4_priori)
    d13c_othfrac.append(d13c_priori)
    
f,ax2 = plt.subplots( 1, sharex = True, figsize = (15,10) )
ax2.set_title('Test for sensitivity to fractionation by the other sink of CH4 \n'+ \
             r'The fractionation constant $\epsilon$ was varied, as indicated in the legend')
ax2.set_ylabel(r'$\delta^{13}$C (permil)')
ax2.set_xlabel('Year')
ax2.plot(range(styear,edyear),d13c_obs,'o',color = obs_color,label= 'Observed')
for i in range(len(epsis)-1,-1,-1):
    ax2.plot(range(styear,edyear),d13c_othfrac[i],markeri[i],color = colori[i],label=str(-epsis[i]))
lgd = ax2.legend(bbox_to_anchor = (1.18, 1.05))
plt.savefig('Sensitivity_test_CH4other_frac',bbox_extra_artists=(lgd,), bbox_inches='tight')

a_ch4_other = a_ch4_other0



























