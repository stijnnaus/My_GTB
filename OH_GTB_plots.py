# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:30:54 2017

@author: naus010

The file in which I can combine experiments, saved from OH_GTB.
I should run OH_GTB_read_data_helper, then OH_GBT, then this file
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt



def read_full(exp_name, dataset):
    '''Reads the 'full' array (opt C and E and OH)'''
    filename = exp_name +'_'+ dataset + '_full'
    filepointer = os.path.join('Data output','arrays',filename)
    full_array = np.loadtxt(filepointer)
    return unpack_full(full_array)
    
def read_x(exp_name, dataset):
    '''Reads the xopt array'''
    filename = exp_name +'_'+ dataset + '_xopt'
    filepointer = os.path.join('Data output','arrays',filename)
    xopt = np.loadtxt(filepointer)
    return xopt

def unpack_full(farray):
    '''Returns the respective components of the full array'''
    yrs = farray[:,0]
    mcf,ch4,d13c = farray[:,1],farray[:,2],farray[:,3]
    fst,fsl,fme = farray[:,4],farray[:,5],farray[:,6]
    fch4,fr13 = farray[:,7],farray[:,8]
    return yrs,(mcf,ch4,d13c,fst,fsl,fme,fch4,fr13)

def mcf_emi(array):
    '''Calculates MCF emissions from an unpacked array'''
    emcf = em0_mcf + mcf_shift(array[3],array[4],array[5])
    return emcf

def reldev_mcf(array):
    '''Computes the relative emission deviations for MCF from unpacked array'''
    emcf = mcf_emi(array)
    return emcf/em0_mcf - 1
    
def combine_xopts(xopts):
    '''
    Combines the unpacked xopts, grouping the different components of
    the state.
    '''
    mcfis,ch4is,r13is,fohs,fsts,fsls,fmes,fch4s,r13es = [],[],[],[],[],[],[],[],[]
    for xopt in xopts:
        mcfi,ch4i,r13i,foh,fst,fsl,fme,fch4,r13e = unpack(xopt)
        mcfis.append(mcfi); ch4is.append(ch4i); r13is.append(r13i)
        fohs.append(foh); fch4s.append(fch4); r13es.append(r13e)
        fsts.append(fst); fsls.append(fsl); fmes.append(fme)
    return mcfis,ch4is,r13is,fohs,fsts,fsls,fmes,fch4s,r13es
    
def fig_reldev(xopts,labels,plottit,legtit,figname=None):
    '''
    Plots the deviations in OH, MCF emissions, CH4 emissions and 
    d13C in CH4 emissions for xopts. Deviations are relative to the prior.
    '''
    nexp = len(xopts)
    _,_,_,foh,fst,fsl,fme,fch4,r13e = combine_xopts(xopts)
    d13c = r13_to_d13c(r13e)
    fig = plt.figure(figsize=(10,100))
    ax1 = fig.add_subplot(411); ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413); ax4 = fig.add_subplot(414)
    ax1.grid(True); ax2.grid(True); ax3.grid(True); ax4.grid(True)
    ax1.set_title(plottit+'\n\nRelative deviations in OH')
    ax2.set_title('Relative deviations in MCF emissions')
    ax3.set_title('Relative deviations in CH4 emissions')
    ax4.set_title('Relative deviations in '+r'$\delta^{13}$C of CH$_4$ emissions')
    ax1.set_ylabel('Deviation (%)'); ax2.set_ylabel('Deviation (%)')
    ax3.set_ylabel('Deviation (%)'); ax4.set_ylabel('Deviation (%)')
    ax4.set_xlabel('Year')
    for i in range(nexp):
        mcfdev = mcf_shift(fst[i],fsl[i],fme[i])/em0_mcf
        ax1.plot(yrs, 100*(foh[i]-1), 'o-', color=sim_blu[i], label=labels[i])
        ax2.plot(yrs, 100*mcfdev, 'o-', color=sim_blu[i], label=labels[i])
        ax3.plot(yrs, 100*(fch4[i]-1), 'o-', color=sim_blu[i], label=labels[i])
        ax4.plot(yrs, d13c[i]-em0_d13c, 'o-', color=sim_blu[i], label=labels[i])
    lgd=ax2.legend(bbox_to_anchor=(1.22,1.),title=legtit)
    fig.tight_layout()
    if figname != None:
        plt.savefig(figloc+'\\'+figname,bbox_extra_artists=(lgd,), bbox_inches='tight')

dif_col = ['blue','black','red','indigo','steelblue','maroon','cyan'] # Very different colors
sim_blu = ['black','navy','blue','steelblue','lightsteelblue'] # Similar colors, going from dark to light
sim_red = ['maroon', 'firebrick','red','lightcoral', 'yellow']
figloc = os.path.join(os.getcwd(), 'Figures')
yrs = np.arange(1992,2015)+.5
d13c_lab =r'$\delta^{13}$C (permil)'
ch4e_lab = r'CH$_4$ emissions (Tg/yr)'
mcfe_lab = 'MCF emissions (Gg/yr)'
ch4o_lab = r'CH$_4$ mixing ratio (ppb)'
mcfo_lab = 'MCF mixing ratio (ppt)'
oh_lab = 'OH concentration \n'+r'($10^6$ molec cm$^{-3}$)'


# Comparing constraints on mcf emissions (experiment name 'em')
x_emcf00 = read_x('emcf00', 'noaa')
x_emcf01 = read_x('emcf01', 'noaa')
x_emcf02 = read_x('emcf02', 'noaa')
x_emcf03 = read_x('emcf03', 'noaa')
x_emcf04 = read_x('emcf04', 'noaa')
x_ems = [x_emcf00, x_emcf01, x_emcf02, x_emcf03, x_emcf04]
lab_em = ['0%','1%','2%','3%','4%'] # label per experiment
leg_em = 'Error in MCF \nemission factors' # Legend title
title_em = 'The effect of varying the constraint on the MCF emission factors' # Plot title
name_em = 'var_fmcfe.png' # Figure name
# Drawing&saving the figure:
fig_reldev(x_ems,lab_em,title_em,leg_em,name_em)

# Influence CH4 correlation length
x_ch4len1 = read_x('ch4len1', 'noaa')
x_ch4len3 = read_x('ch4len3', 'noaa')
x_ch4len6 = read_x('ch4len6', 'noaa')
x_ch4len10 = read_x('ch4len10', 'noaa')
x_lens = [x_ch4len1,x_ch4len3,x_ch4len6,x_ch4len10]
lab_lens = ['1','3','6','10'] # label per experiment
leg_lens = 'Correlation length (yr)'
title_lens = 'The influence of CH4 correlation length on derived CH4 and CH4 emissions'
name_lens = 'var_ch4len.png'

_,_,_,foh_lens,fst_lens,fsl_lens,fme_lens,fch4_lens,r13e_lens = combine_xopts(x_lens)
fig = plt.figure(figsize=(10,80))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
ax1.set_title(r'CH$_4$ concentrations and $\delta^{13}$C concentrations:'+' both from\n observations and modelled forward from the prior.')
ax1.set_ylabel(ch4o_lab)
ax2.set_ylabel(d13c_lab)
ax3.set_ylabel(ch4e_lab)
ax4.set_ylabel(oh_lab)
ax4.set_xlabel('Year')
ax1.errorbar(yrs,ch4_obs,yerr=ch4_obs_e,fmt='go',label=r'CH$_4$, obs')
ax2.errorbar(yrs,d13c_obs, yerr=d13c_obs_e,fmt = 'go',label=r'$\delta^{13}$C, obs')
for i,x in enumerate(x_lens):
    ch4m,r13m,_ = forward_ch4(x)
    d13cm = r13_to_d13c(r13m)
    ax1.plot(yrs,ch4m,'-',color=sim_blu[i],label=r'CH$_4$, opt, '+lab_lens[i])
    ax2.plot(yrs,d13cm,'-',color=sim_blu[i],label=r'$\delta^{13}$C, opt, '+lab_lens[i])
    ax3.plot(yrs,fch4_lens[i]*em0_ch4,'o-',color=sim_blu[i])
    ax4.plot(yrs,foh_lens[i]*oh/1e6,'o-',color=sim_blu[i])
lgd=ax2.legend(bbox_to_anchor=(1.22,1.),title=leg_lens)
fig.tight_layout()
plt.savefig(figloc+'\\'+'name_lens',bbox_extra_artists=(lgd,), bbox_inches='tight')

# What happens if I don't optimize one of the 3
x_normal = read_x('normal', 'noaa')
x_noch4 = read_x('offch4', 'noaa')
x_nomcf = read_x('offmcf', 'noaa')
x_nooh = read_x('offoh', 'noaa')
x_onoff = [x_normal, x_noch4, x_nomcf, x_nooh]
lab_onoff = ['normal','no ch4','no mcf','no oh'] # label per experiment
leg_onoff = '' # Legend title
title_onoff = 'The effect of excluding one of the parameters from the optimization' # Plot title
name_onoff = 'onoff.png' # Figure name
# Drawing&saving the figure:
fig_reldev(x_onoff,lab_onoff,title_onoff,leg_onoff,name_onoff)


































