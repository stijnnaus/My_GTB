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

reduc = 1e0
def calculate_J(xp):
    x = precon_to_state(xp)
    _, mismatch = calc_mismatch(x)
    J_pri = .5 * sum(dot(b_inv, (x - x_prior)**2)) # prior
    J_obs = .5 * sum( (mismatch / con_data_e)**2 ) # mismatch with obs
    J_tot = J_pri + J_obs
    print 'Cost function value:',J_tot*reduc
    return J_tot*reduc
    
def calculate_dJdx(xp):
    x = precon_to_state(xp)
    foh = x[:nt]
    
    MCF_save,C12H4_save,C13H4_save = forward_all(x)
    adj_mcf = adjoint_model_mcf(foh, MCF_save)
    adj_c12 = adjoint_model_c12(foh, C12H4_save)
    adj_c13 = adjoint_model_c13(foh, C13H4_save)
    dfoh = adj_mcf[0] + adj_c12[0] + adj_c13[0]
    dJdx_obs = np.concatenate((dfoh, adj_mcf[1],adj_mcf[2], \
                                adj_c12[1],adj_c12[2],adj_c13[1],adj_c13[2]))
                                
    dJdx_pri = dot(b_inv, (x-x_prior))
    dJdx = dJdx_obs + dJdx_pri
    dJdxp = dot( L_adj, dJdx )
    print 'Cost function deriv:',max(dJdxp)*reduc
    return dJdxp*reduc

def adjoint_model_mcf( foh, MCF_save ):
    mismatch = ( MCF_save - mcf_obs )
    pulse_MCF = adj_obs_oper( mismatch / mcf_obs_e**2 )
    dMCF= zeros(nt)
    dfmcf,dfoh = zeros(nt),zeros(nt)
    dMCFi = 0
    
    rapidc = rapid/conv_mcf
    
    for i in range(edyear-1,styear-1,-1):
        iyear = i-styear
        
        # Add adjoint pulses
        dMCFi   += pulse_MCF[iyear]
        
        # Chemistry
        dMCFi   = dMCFi *  (1 - foh[iyear] * l_mcf_oh - l_mcf_strat - l_mcf_ocean)
        dMCF[iyear] = dMCFi
        dfoh[iyear] = - l_mcf_oh * MCF_save[iyear] * dMCF[iyear]
                        
        # Emissions
        dstock = 0.
        for j in range(i+1 , i+11):
            jyear = j - styear
            if j < edyear: dstock += 0.1 * rapidc[iyear] * dMCF[jyear]
        dfmcf[iyear] = dfmcf[iyear] - 0.75 * rapidc[iyear] * dMCF[iyear]  + dstock
        if (i + 1) < edyear: dfmcf[iyear] -= -0.25 * rapidc[iyear] * dMCF[iyear+1]
            
    adj_mcf = [dfoh, array([dMCFi]), dfmcf]
    return adj_mcf
    
def adjoint_model_c12( foh, C12H4_save ):
    mismatch = ( C12H4_save - c12_obs )
    pulse_12CH4 = adj_obs_oper( mismatch / c12_obs_e**2 )
    em0_12 = em0_c12 / conv_ch4
    
    d12CH4 = zeros(nt)
    df12,dfoh = zeros(nt),zeros(nt)
    d12CH4i = 0
    
    for i in range(edyear-1, styear-1, -1):
        iyear = i-styear
        d12CH4i += pulse_12CH4[iyear]
        
        d12CH4i = d12CH4i * (1 - foh[iyear] * l_ch4_oh - l_ch4_other)
        d12CH4[iyear] = d12CH4i
        dfoh[iyear] = - l_ch4_oh * C12H4_save[iyear] * d12CH4[iyear]
        
        df12[iyear] = em0_12[iyear] * d12CH4[iyear]
        
    adj_c12 = [dfoh, array([d12CH4i]), df12]
    return adj_c12
    
def adjoint_model_c13( foh, C13H4_save ):
    mismatch = ( C13H4_save - c13_obs )
    pulse_13CH4 = adj_obs_oper( mismatch / c13_obs_e**2 )
    
    d13CH4 = zeros(nt)
    df13,dfoh = zeros(nt),zeros(nt)
    d13CH4i = 0
    em0_13 = em0_c13 / conv_13ch4
    
    for i in range(edyear-1, styear-1, -1):
        iyear = i-styear
        d13CH4i += pulse_13CH4[iyear]
        
        d13CH4i = d13CH4i * (1 - foh[iyear] * l_ch4_oh * a_ch4_oh - a_ch4_other * l_ch4_other)
        d13CH4[iyear] = d13CH4i
        dfoh[iyear] = - l_ch4_oh * C13H4_save[iyear] * d13CH4[iyear] * a_ch4_oh
        
        df13[iyear] = em0_13[iyear] * d13CH4[iyear]
        
    adj_c13 = [dfoh, array([d13CH4i]), df13]
    return adj_c13
    
def calc_mismatch(x):
    C = forward_all(x)
    C_obs = obs_oper(C)
    return C,array(C_obs - con_data)

def forward_all(x):
    C_mcf,C_12ch4,C_13ch4 = forward_mcf(x),forward_12ch4(x),forward_13ch4(x)
    return C_mcf,C_12ch4,C_13ch4

def forward_mcf(x):
    f_oh, mcf0, f_mcf = x[:nt], x[nt], x[nt+1:2*nt+1]
    em = em0_mcf + mcf_shift(f_mcf)
    em /= conv_mcf
    
    mcfs = []; mcf = mcf0
    for year in range(styear,edyear):
        i = year - styear
        mcf += em[i]
        mcf = mcf * ( 1 - l_mcf_ocean - l_mcf_strat - l_mcf_oh * f_oh[i] )
        mcfs.append(mcf)
        
    return array(mcfs)
    
def forward_12ch4(x):
    f_oh, C12H4_0, f_C12H4 = x[:nt], x[2*nt+1], x[2*nt+2 : 3*nt+2]
    em0_12 = em0_c12 / conv_ch4
    
    C12H4s = []; C12H4 = C12H4_0
    for year in range(styear,edyear):
        i = year - styear
        C12H4 += f_C12H4[i] * em0_12[i]
        C12H4  = C12H4 * ( 1 - l_ch4_other - l_ch4_oh * f_oh[i])
        C12H4s.append(C12H4)
    
    return array(C12H4s)
   
def forward_13ch4(x):
    f_oh, C13H4_0, f_C13H4 = x[:nt], x[3*nt+2], x[3*nt+3 : 4*nt+3]
    em0_13 = em0_c13 / conv_13ch4
    
    C13H4s = []; C13H4 = C13H4_0
    for year in range(styear,edyear):
        i = year - styear
        C13H4 += f_C13H4[i] * em0_13[i]
        C13H4  = C13H4 * ( 1 - a_ch4_other * l_ch4_other - a_ch4_oh * l_ch4_oh * f_oh[i] )
        C13H4s.append(C13H4)
    
    return array(C13H4s)

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
            fyear = yearb - styear   
            if fyear >= 0: shift += 0.1*f_mcf[fyear]*rapid[eyear]
        shifts.append(shift)
    
    return array(shifts)
    
def obs_oper(con):
    return con

def adj_obs_oper(con):
    return con
    
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

def split_to_deltot(C12H4, C13H4):
    '''
    Converts split 12CH4 and 13CH4 values (emission/concentration)  
    to a delta 13C value and a total CH4 quantity.
    '''
    R_sample = C13H4 / C12H4
    delCH4 = ( R_sample / R_ST ) - 1
    totCH4 = C12H4 + C13H4
    return delCH4, totCH4

def deltot_to_split(totCH4, delc, totCH4_e = [None], delc_e = [None]):
    '''
    Converts a (list of) d13C and total CH4 values to 12CH4 and 13CH4.
    '''
    if type(delc) == float:
        q = R_ST * ( delc/1000. + 1 )
        C12H4 = totCH4 / (1+q)
        C13H4 = q * C12H4
        return C12H4, C13H4
        
    totCH4, delc = array(totCH4), array(delc)/1000.
    q = array( R_ST * (delc+1) )
    C12H4 = totCH4 / (q+1)
    C13H4 = q*C12H4
    
    if delc_e[0] == None or totCH4_e[0] == None:
        return C12H4, C13H4
    else: 
        totCH4_e, delc_e = array(totCH4_e), array(delc_e)/1000.
        C12H4_e = sqrt((   totCH4_e / (1+q) )**2 + \
                    ( delc_e * totCH4 * R_ST / (1+q)**2 )**2)
        C13H4_e = sqrt(( q*totCH4_e / (1+q) )**2 + \
                    ( delc_e * totCH4 * R_ST * (1/(1+q) - q/(1+q)**2) )**2)
        return C12H4, C13H4, C12H4_e, C13H4_e
    

# setup model variables, emissions, etc:
m = 5e18
xmair = 28.5
xmcf = 133.5
xch4 = 16.0
x13ch4 = 17.0
conv_mcf = xmcf / 10**12 * m / xmair # kg/ppt
conv_ch4 = xch4 / 10**9  * m / xmair # kg/ppb
conv_13ch4 = x13ch4 / 10**9  * m / xmair # kg/ppb
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
em0_ch4 = array([540.0]*nt)*1e9
d13c_obs,d13c_obs_e = read_d13C_obs(os.path.join('OBSERVATIONS','d13C_Schaefer.txt'))
em0_d13c = array([-53.5]*nt)
c12_obs, c13_obs, c12_obs_e, c13_obs_e = deltot_to_split(ch4_obs,d13c_obs,ch4_obs_e,d13c_obs_e)
em0_c12, em0_c13 = deltot_to_split(em0_ch4, em0_d13c)
con_data = array([mcf_obs,c12_obs,c13_obs])
con_data_e = array([mcf_obs_e,c12_obs_e,c13_obs_e])

mcf_ini = array([mcf_obs[0]])
c12_ini = array([c12_obs[0]])
c13_ini = array([c13_obs[0]])
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
xp_opt = optimize.fmin_bfgs(calculate_J,xp_prior,calculate_dJdx, gtol=1e-1)
x_opt = precon_to_state(xp_opt)

plt.figure()
plt.plot(x_prior)
plt.plot(x_opt)



'''
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_ylabel(r'$^{12}$CH$_4$ concentrations (ppb)')
ax2.set_ylabel(r'$^{13}$CH$_4$ concentrations (ppb)')
ax1.set_xlabel('Year')
ax1.plot(range(styear,edyear),c12_obs,'o', color = 'lightblue', label = r'$^{12}$CH$_4$ obs')
ax1.plot(range(styear,edyear),forward_12ch4(x_prior),'-', color = 'lightblue', label = r'$^{12}$CH$_4$ prior')
ax2.plot(range(styear,edyear),c13_obs,'o', color = 'blue', label = r'$^{13}$CH$_4$ obs')
ax2.plot(range(styear,edyear),forward_13ch4(x_prior),'-', color = 'blue', label = r'$^{13}$CH$_4$ model')
ax1.legend(loc = 'upper left')
ax2.legend(loc = 'lower right')

plt.figure()
plt.xlabel('Year')
plt.ylabel('MCF concentration (ppb)')
plt.plot(range(styear,edyear),mcf_obs,'o', color = 'gray',label = 'MCF obs')
plt.plot(range(styear,edyear),forward_mcf(x_prior),'ro', label = 'MCF prior')
plt.legend(loc='best')

fig_em = plt.figure()
ax1 = fig_em.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_ylabel(r'$^{12}$CH$_4$ emissions (Tg/yr)')
ax2.set_ylabel(r'$^{13}$CH$_4$ emissions (Tg/yr)')
ax1.set_xlabel('Year')
ax1.plot(range(styear,edyear),em0_c12,'o', color = 'lightblue', label = r'$^{12}$CH$_4$')
ax2.plot(range(styear,edyear),em0_c13,'o', color = 'blue',      label = r'$^{13}$CH$_4$')
ax1.legend(loc='upper left')
ax2.legend(loc='lower right')


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_ylabel(r'concentrations (ppb)')
ax2.set_ylabel(r'errors (ppb)')
ax1.set_xlabel('Year')
ax1.plot(range(styear,edyear),c12_obs,'o', color = 'blue', label = r'$^{12}$CH$_4$ obs')
ax2.plot(range(styear,edyear),c12_obs_e,'v',color = 'blue',label = r'$^{12}$CH$_4$ errors')
ax1.plot(range(styear,edyear),ch4_obs,'o', color = 'gray', label = r'CH$_4$ obs')
ax2.plot(range(styear,edyear),ch4_obs_e,'v',color = 'gray',label = r'CH$_4$ errors')
ax1.legend(loc = 'upper left')
ax2.legend(loc = 'lower right')


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_ylabel(r'CH$_4$ concentrations (ppb)')
ax2.set_ylabel(r'CH$_4$ errors (ppb)')
ax1.set_xlabel('Year')
ax1.plot(range(styear,edyear),c13_obs,'o', color = 'lightblue', label = r'$^{13}$CH$_4$ obs')
ax2.plot(range(styear,edyear),c13_obs_e,'v',color = 'blue',label = r'$^{13}$CH$_4$ errors')
ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper right')
'''



# Adjoint test

# MCF adjoint test
x1 = 10+20*np.random.rand(nstate)
Mx = forward_mcf(x1)
x1 = np.concatenate((x1[:nt],array([x1[nt]]),x1[nt+1:2*nt+1]))

y = 100 + 100*np.random.rand(nt)

foh_test = x1[:nt]
MTy = adjoint_model_mcf(foh_test, y)
MTy = np.concatenate((MTy[0],MTy[1],MTy[2]))

print np.dot(Mx,(y-mcf_obs)/mcf_obs_e**2), np.dot(x1,MTy)

# 12CH4 adjoint test

x1 = 10+20*np.random.rand(nstate)
Mx = forward_12ch4(x1)
x1 = np.concatenate((x1[:nt],array([x1[2*nt+1]]),x1[2*nt+2:3*nt+2]))

y = 100 + 100*np.random.rand(nt)

foh_test = x1[:nt]
MTy = adjoint_model_c12(foh_test, y)
MTy = np.concatenate((MTy[0],MTy[1],MTy[2]))

print np.dot(Mx,y), np.dot(x1,MTy)

# Gradient test

def grad_test(x0,pert = 10**(-5)):
    nx = len(x0)/2
    x0 = np.array(x0)
    x_priorA = x0
    deriv = calc_dJdx(x0)
    dE,dC = deriv[:nx], deriv[nx:]
    J_prior = calc_J(x0)
    
    values = []
    for i in range(nx):
        pert_array = np.zeros(2*nx)
        pert_array[i] = pert
        x0_pert = x0 + pert_array
        
        predict = pert*deriv[i]
        J_post = calc_J(x0_pert)
        reduct = (J_post - J_prior)
        if predict == reduct == 0:
            val = 0
        else:
            val = abs((predict - reduct)/( ( predict + reduct ) / 2 ))
#        print 'For grid cell',i,'......'
#        if val <= 0.01:
#            print 'Gradient test passed :)'
#        else:
#            print 'Gradient test failed :('
        values.append(val)
    return np.array( values )*100



