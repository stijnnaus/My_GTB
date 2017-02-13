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

plt.rcParams.update({'font.size': 16})

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
    
def fig_reldev(xopts,labels,plottit,legtit=None,figname=None):
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
        ax1.plot(yrs, 100*(foh[i]-1), 'o-', color=dif_col[i], label=labels[i])
        ax2.plot(yrs, 100*mcfdev, 'o-', color=dif_col[i], label=labels[i])
        ax3.plot(yrs, 100*(fch4[i]-1), 'o-', color=dif_col[i], label=labels[i])
        ax4.plot(yrs, d13c[i]-em0_d13c, 'o-', color=dif_col[i], label=labels[i])
    lgd=ax2.legend(bbox_to_anchor=(1.22,1.),title=legtit)
    fig.tight_layout()
    if figname==None: figname='default'
    plt.savefig(figloc+'\\'+figname,bbox_extra_artists=(lgd,), bbox_inches='tight')
        
def fig_mcfch4_obs(xopts,labels,plottit,legtit=None,dataset=None,figname=None):
    '''
    Plots the MCF, CH4C observations, along with the model results
    from xopts.
    Dataset: Which dataset will be plotted. Either 'noaa','agage' or 'both'
    ''' 
    nexp = len(xopts)
    fig = plt.figure(figsize=(10,50))
    ax1 = fig.add_subplot(211) # MCF observations
    ax2 = fig.add_subplot(212) # CH4 observations
    ax1.set_title(plottit+'\n\nMethyl chloroform')
    ax2.set_title(r'Methane')
    ax1.set_ylabel('MCF (ppt)'); ax2.set_ylabel(r'CH$_4$ (ppb)')
    ax2.set_xlabel('Year')
    if dataset=='noaa' or dataset=='both':
        ax1.errorbar(yrs, mcf_noaa, yerr=mcf_obs_e, fmt='o', color='g',label='NOAA obs')
        ax2.errorbar(yrs, ch4_noaa, yerr=ch4_obs_e, fmt='o', color='g',label='NOAA obs')
    if dataset=='agage' or dataset=='both':
        ax1.errorbar(yrs, mcf_agage, yerr=mcf_obs_e, fmt='o', color='r',label='AGAGE obs')
        ax2.errorbar(yrs, ch4_agage, yerr=ch4_obs_e, fmt='o', color='r',label='AGAGE obs')
    for i in range(nexp):
        mcfi = forward_mcf(xopts[i])
        ch4i,_,_ = forward_ch4(xopts[i])
        ax1.plot(yrs, mcfi, '-', color=dif_col[i], label=labels[i]+' opt')
        ax2.plot(yrs, ch4i, '-', color=dif_col[i], label=labels[i]+' opt')
    lgd=ax2.legend(bbox_to_anchor=(1.3,1.),title=legtit)
    fig.tight_layout()
    if figname==None: figname='default'
    plt.savefig(figloc+'\\'+figname,bbox_extra_artists=(lgd,), bbox_inches='tight')
    
def fig_all_obs(xopts,labels,plottit,legtit=None,dataset=None,figname=None):
    '''
    Plots the MCF, CH4 and d13C observations, along with the model results
    from xopts.
    Dataset: Which dataset will be plotted. Either 'noaa','agage' or 'both'
    ''' 
    nexp = len(xopts)
    fig = plt.figure(figsize=(10,50))
    ax1 = fig.add_subplot(311) # MCF observations
    ax2 = fig.add_subplot(312) # CH4 observations
    ax3 = fig.add_subplot(313) # d13C observations
    ax1.set_title(plottit+'\n\nMethyl chloroform')
    ax2.set_title(r'Methane')
    ax3.set_title(r'$\delta^{13}$C in CH$_4$')
    ax1.set_ylabel('MCF (ppt)'); ax2.set_ylabel(r'CH$_4$ (ppb)')
    ax3.set_ylabel(r'$\delta^{13}$C (permil)')
    ax3.set_xlabel('Year')
    if dataset=='noaa' or dataset=='both':
        ax1.errorbar(yrs, mcf_noaa, yerr=mcf_obs_e, fmt='o', color='g',label='NOAA obs')
        ax2.errorbar(yrs, ch4_noaa, yerr=ch4_obs_e, fmt='o', color='g',label='NOAA obs')
    if dataset=='agage' or dataset=='both':
        ax1.errorbar(yrs, mcf_agage, yerr=mcf_obs_e, fmt='o', color='r',label='AGAGE obs')
        ax2.errorbar(yrs, ch4_agage, yerr=ch4_obs_e, fmt='o', color='r',label='AGAGE obs')
    ax3.errorbar(yrs, d13c_obs, yerr=d13c_obs_e, fmt='o', color='g', label='obs')
    for i in range(nexp):
        mcfi = forward_mcf(xopts[i])
        ch4i,r13i,_ = forward_ch4(xopts[i])
        d13ci = r13_to_d13c(r13i)
        ax1.plot(yrs, mcfi, '-', color=dif_col[i], label=labels[i]+' opt')
        ax2.plot(yrs, ch4i, '-', color=dif_col[i], label=labels[i]+' opt')
        ax3.plot(yrs, d13ci,'-', color=dif_col[i], label=labels[i]+' opt')
    lgd=ax2.legend(bbox_to_anchor=(1.3,1.),title=legtit)
    fig.tight_layout()
    if figname==None: figname='default'
    plt.savefig(figloc+'\\'+figname,bbox_extra_artists=(lgd,), bbox_inches='tight')
    
def fig_oh_v_ch4growth(xopts,labels,plottit,legtit=None,dataset='noaa',figname=None):
    '''
    Plots the CH4 growth rate versus OH concentrations.
    '''
    if dataset=='noaa':
        ch4_growth = [ch4_noaa[i]-ch4_noaa[i-1] for i in range(1,nt)]
    elif dataset=='agage':
        ch4_growth = [ch4_agage[i]-ch4_agage[i-1] for i in range(1,nt)]
    else: print 'Select a valid dataset: NOAA or AGAGE!'
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(plottit+'\n\n The correlation between the OH and the CH4 growth rate')
    ax1.set_xlabel('CH4 growth rate (ppb/yr)')
    ax1.set_ylabel(oh_lab)
    for i,xopt in enumerate(xopts):
        _,_,_,fohi,_,_,_,_,_ = unpack(xopt)
        ohu = fohi*oh
        oh_mid = array([(ohu[j]+ohu[j-1])/2 for j in range(1,nt)])
        cor = round(np.corrcoef(ch4_growth,oh_mid)[0,1],2)
        ax1.plot(ch4_growth, oh_mid/1e6, 'o',color=dif_col[i], label=labels[i]+'(r='+str(cor)+')')
    lgd=ax1.legend(bbox_to_anchor=(1.24,1.),title=legtit)
    fig.tight_layout()
    if figname==None: figname='default'
    plt.savefig(figloc+'\\'+figname,bbox_extra_artists=(lgd,), bbox_inches='tight')
    
def lifetime_plot(xopts,labels,plottit,legtit=None ,figname=None):
    '''
    Plot of MCF and CH4 lifetimes against OH and the total lifetime.
    '''
    fig = plt.figure(figsize=(10,20))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.set_title(plottit+'\n\n The lifetime of MCF against OH and in total')
    ax2.set_title(r'The lifetime of CH$_4$ against OH and in total')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Lifetime (years)')
    ax2.set_ylabel('Lifetime (years)')
    for i,xopt in enumerate(xopts):
        _,_,_,fohi,_,_,_,_,_ = unpack(xopt)
        loh = fohi*l_mcf_oh
        tau_mcf_oh = 1/loh
        tau_mcf_tot = 1/(loh+l_mcf_strat+l_mcf_ocean)
        loh = fohi*l_ch4_oh
        tau_ch4_oh = 1/loh
        tau_ch4_tot = 1/(loh+l_ch4_other)
        ax1.plot(yrs, tau_mcf_oh, 'o',color=dif_col[i],label=labels[i]+' v OH')
        ax1.plot(yrs, tau_mcf_tot, 'v',color=dif_col[-i],label=labels[i]+' total')
        ax2.plot(yrs, tau_ch4_oh, 'o',color=dif_col[i],label=labels[i]+' v OH')
        ax2.plot(yrs, tau_ch4_tot, 'v',color=dif_col[-i],label=labels[i]+' total')
    ax1.legend(loc='best')
    plt.tight_layout()
    if figname == None: figname='default'
    plt.savefig(figname)
    
def mid(arr):
    '''Returns the linearly interpolated values between array values '''
    return (arr[1:]+arr[:-1])/2.
    
dif_col = ['blue','maroon','steelblue','pink','black','cyan','red','firebrick'] # Very different colors
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
ax1.set_title(r'The effect of varying the correlation length of CH$_4$ emissions.')
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
    ax2.plot(yrs,d13cm,'-',color=sim_blu[i],label=lab_lens[i]+r', opt')
    ax3.plot(yrs,fch4_lens[i]*em0_ch4,'o-',color=sim_blu[i])
    ax4.plot(yrs,foh_lens[i]*oh/1e6,'o-',color=sim_blu[i])
lgd=ax2.legend(bbox_to_anchor=(1.26,1.),title=leg_lens)
fig.tight_layout()
plt.savefig(figloc+'\\'+'ch4_corlens',bbox_extra_artists=(lgd,), bbox_inches='tight')

# What happens if I don't optimize one of the 3
x_normal = read_x('normal', 'noaa')
x_noch4 = read_x('offch4', 'noaa')
x_nomcf = read_x('offmcf', 'noaa')
x_nooh = read_x('offoh', 'noaa')
x_onoff = [x_normal, x_noch4, x_nomcf, x_nooh]
lab_onoff = ['normal','no ch4','no mcf','no oh'] # label per experiment
title_onoff = 'The effect of excluding one of the parameters from the optimization' # Plot title
name_onoff = 'onoff_reldev.png' # Figure name
name_onoff2 = 'onoff_obs.png'
fig_reldev(x_onoff,lab_onoff,title_onoff,figname=name_onoff)
fig_all_obs(x_onoff,lab_onoff,title_onoff,dataset='noaa',figname=name_onoff2)

# NOAA vs AGAGE
x_noaa = read_x('normal', 'noaa')
x_agag = read_x('normal', 'agage')
x_sets = [x_noaa,x_agag]
lab_sets = ['NOAA', 'AGAGE']
name_sets = 'dataset_comp_mcf_ch4.png'
tit_sets = 'Comparison of observations from the two \n\
    datasets, and their optimized results'
fig_mcfch4_obs(x_sets, lab_sets, tit_sets, dataset='both',figname=name_sets)
name_sets2 = 'dataset_comp_reldev.png'
tit_sets2 = 'Comparison of the optimized states from the two datasets'
fig_reldev(x_sets,lab_sets, tit_sets2, figname=name_sets2)

# Varying MCF obs error
x_mcfe05 = read_x('mcfe0.5','noaa')
x_mcfe10 = read_x('normal','noaa')
x_mcfe15 = read_x('mcfe1.5','noaa')
x_mcfe20 = read_x('mcfe2.0','noaa')
x_mcfe25 = read_x('mcfe2.5','noaa')
x_mcfe = [x_mcfe05,x_mcfe10,x_mcfe15,x_mcfe20,x_mcfe25]
lab_mcfe = ['x0.5','x1.0','x1.5','x2.0','x2.5']
name_mcfe = 'varying_mcf_obs_e.png'
leg_mcfe = 'Multiplication \n factor'
tit_mcfe = 'The effect of varying the observation error in MCF'
fig_reldev(x_mcfe,lab_mcfe,tit_mcfe,legtit=leg_mcfe,figname=name_mcfe)

# Varying the entire prior error simultaneously
x_pri10 = read_x('prie1.0','noaa')
x_pri06 = read_x('prie0.6','noaa')
x_pri04 = read_x('prie0.4','noaa')
x_pri02 = read_x('prie0.2','noaa')
x_prie = [x_pri10,x_pri06,x_pri04,x_pri02]
lab_prie = ['x1.0','x0.6','x0.4','x0.2']
name_prie = 'vary_prior_e_reldev.png'
name_prie2 = 'vary_prior_e_obs.png'
leg_prie = 'Multiplication \n factor'
tit_prie = 'The effect of varying the prior errors'
fig_reldev(x_prie,lab_prie,tit_prie,legtit=leg_prie,figname=name_prie)
fig_all_obs(x_prie, lab_prie, tit_prie, dataset='noaa',figname=name_prie2)

# Growth rate CH4 and OH correlation plot
title_cor = 'Influence of prior error on the strength of the correlation'
lab_cor = ['normal','prierr*0.6','prierr*0.2']
name_cor = 'CH4growth_vs_OH_prierr.png'
x_cor = [x_pri10,x_pri06,x_pri02]
fig_oh_v_ch4growth(x_cor,lab_cor,title_cor,figname=name_cor)
x_cor2 = [x_noaa,x_agag]
name_cor2 = 'CH4growth_vs_OH_datasets.png'
lab_cor2 = ['NOAA','AGAGE']
title_cor2 = 'Difference between NOAA and AGAGE'
fig_oh_v_ch4growth(x_cor2,lab_cor2,title_cor2,figname=name_cor2)

# Overfitting
x_overn = read_x('overfit','noaa')
x_overa = read_x('overfit','agage')
labels = ['Overfit noaa', 'Overfit agage']
fig_reldev([x_overn,x_overa],labels,'',figname='Overfitting.png')

# Lifetime plots NOAA v AGAGE
lifetime_plot([x_noaa,x_agag],['NOAA','AGAGE'],'The lifetime of MCF and CH4',figname='lifetime noaa v agage.png')

# Default normal obs plot
fig_all_obs([x_noaa],['NOAA'],'',dataset='noaa',figname='standard_obs_plot.png')

# Low correlation length Methane and OH
x_low = read_x('lowcor','noaa')
x_lowr = read_x('lowercor','noaa')
labs = ['Normal','Lower','Lowest']
titl = 'The effect of reducing the correlation length in CH4 and OH\n\
        from 10 and 1yr to 2/.1 and .1/.1yr respectively'
fig_reldev([x_noaa,x_low,x_lowr],labs,titl,figname='Lowcor_reldev.png')
fig_oh_v_ch4growth([x_noaa,x_low,x_lowr],labs,titl,figname='Lowcor_cor.png')

# Methane growth rates
Tg = 1e-9
figloce = figloc+'\\Emissions'
emcf_opt_noaa = em0_mcf+mcf_shiftx(x_noaa)
emcf_opt_agag = em0_mcf+mcf_shiftx(x_agag)
ch4_growth_noaa = ch4_noaa[1:]-ch4_noaa[:-1]
ch4_growth_agag = ch4_agage[1:]-ch4_agage[:-1]
mcf_growth_noaa = mcf_noaa[1:]-mcf_noaa[:-1]
mcf_growth_agag = mcf_agage[1:]-mcf_agage[:-1]
mcf_mid_noaa = mid(mcf_noaa)
mcf_mid_agag = mid(mcf_agage)
_,_,_,foh_noaa,_,_,_,fch4_noaa,_ = unpack(x_noaa)
_,_,_,foh_agag,_,_,_,fch4_agag,_ = unpack(x_agag)
foh_mid_noaa = mid(foh_noaa)
foh_mid_agag = mid(foh_agag)
ech4_mid_noaa = mid(fch4_noaa) * em0_ch4[0]
ech4_mid_agag = mid(fch4_agag) * em0_ch4[0]
emcf_mid_noaa = mid(emcf_opt_noaa)
emcf_mid_agag = mid(emcf_opt_agag)

cor_ch4_oh_noaa = round(np.corrcoef(ch4_growth_noaa,foh_mid_noaa)[0,1],2)
cor_ch4_oh_agag = round(np.corrcoef(ch4_growth_agag,foh_mid_agag)[0,1],2)
cor_mcf_oh_noaa = round(np.corrcoef(mcf_growth_noaa,foh_mid_noaa)[0,1],2)
cor_mcf_oh_agag = round(np.corrcoef(mcf_growth_agag,foh_mid_agag)[0,1],2)
cor_ch4_em_noaa = round(np.corrcoef(ch4_growth_noaa,ech4_mid_noaa)[0,1],2)
cor_ch4_em_agag = round(np.corrcoef(ch4_growth_agag,ech4_mid_agag)[0,1],2)
cor_mcf_em_noaa = round(np.corrcoef(mcf_growth_noaa,emcf_mid_noaa)[0,1],2)
cor_mcf_em_agag = round(np.corrcoef(mcf_growth_agag,emcf_mid_agag)[0,1],2)
cor_ch4em_oh_noaa = round(np.corrcoef(ech4_mid_noaa,foh_mid_noaa)[0,1],2)
cor_ch4em_oh_agag = round(np.corrcoef(ech4_mid_agag,foh_mid_agag)[0,1],2)

fig=plt.figure(figsize=(10,30))
ax1=fig.add_subplot(311)
ax2=fig.add_subplot(312)
ax3=fig.add_subplot(313)
ax1.set_title('Comparison of correlation CH4 growth rate with OH for NOAA v AGAGE\n\n CH4 growth rate')
ax2.set_title('OH')
ax1.set_ylabel(r'CH$_4$ growth rate (ppb/yr)')
ax2.set_ylabel(r'OH (10$^6$ molec cm$^{-3}$')
ax2.set_xlabel('Years')
ax3.set_xlabel(r'OH (10$^6$ molec cm$^{-3}$')
ax3.set_ylabel(r'CH$_4$ growth rate (ppb/yr)')
ax1.grid(); ax2.grid()
ax1.plot(years[1:],ch4_growth_noaa,'o-',color='steelblue',label='NOAA')
ax1.plot(years[1:],ch4_growth_agag,'o-',color='maroon',label='AGAGE')
ax2.plot(years[1:],foh_mid_noaa*oh*1e-6,'o-',color='steelblue',label='NOAA')
ax2.plot(years[1:],foh_mid_agag*oh*1e-6,'o-',color='maroon',label='AGAGE')
ax3.plot(foh_mid_noaa*oh*1e-6,ch4_growth_noaa,'o',color='steelblue',label='NOAA, r='+str(cor_ch4_oh_noaa))
ax3.plot(foh_mid_agag*oh*1e-6,ch4_growth_agag,'o',color='maroon',label='AGAGE, r='+str(cor_ch4_oh_agag))
ax1.legend(loc='best')
ax3.legend(loc='best')
plt.tight_layout()
plt.savefig(figloce+'\\CH4 growth v OH')

fig = plt.figure(figsize=(10,30))
ax1=fig.add_subplot(311)
ax2=fig.add_subplot(312)
ax3=fig.add_subplot(313)
ax1.set_title(r'CH$_4$ growth vs OH')
ax2.set_title(r'CH$_4$ growth vs CH$_4$ emissions')
ax3.set_title(r'CH$_4$ emissoins vs OH')
ax1.set_ylabel(r'CH$_4$ growth rate (ppb/yr)')
ax3.set_ylabel('MCF growth rate (%)')
ax1.set_xlabel(r'OH (10$^6$ molec cm$^{-3})$')
ax2.set_xlabel(r'CH$_4$ emissions (Tg/yr)')
ax3.set_xlabel(r'OH (10$^6$ molec cm$^{-3}$)')
ax1.grid();ax2.grid();ax3.grid()
ax1.plot(foh_mid_noaa*oh*1e-6,ch4_growth_noaa,'o',color='steelblue',label='NOAA, r='+str(cor_ch4_oh_noaa))
ax1.plot(foh_mid_agag*oh*1e-6,ch4_growth_agag,'o',color='maroon',label='AGAGE, r='+str(cor_ch4_oh_agag))
ax2.plot(ech4_mid_noaa*Tg,ch4_growth_noaa,'o',color='steelblue',label='NOAA, r='+str(cor_ch4_em_noaa))
ax2.plot(ech4_mid_agag*Tg,ch4_growth_agag,'o',color='maroon',label='AGAGE, r='+str(cor_ch4_em_agag))
ax3.plot(ech4_mid_noaa*Tg,foh_mid_noaa*oh*1e-6,'o',color='steelblue',label='NOAA, r='+str(cor_ch4em_oh_noaa))
ax3.plot(ech4_mid_agag*Tg,foh_mid_agag*oh*1e-6,'o',color='maroon',label='AGAGE, r='+str(cor_ch4em_oh_agag))

ax1.legend(loc='best'); ax2.legend(loc='best'); ax3.legend(loc='best')
plt.tight_layout()
plt.savefig(figloce+'\\Correlation plots CH4.png')


fig_oh_v_ch4growth([x_noaa],['noaa'],'Correlation OH with CH4 for noaa data',figname='OH_v_CH4growth_noaa.png',dataset='noaa')
fig_oh_v_ch4growth([x_agag],['agage'],'Correlation OH with CH4 for noaa data',figname='OH_v_CH4growth_agag.png',dataset='agage')

# MCF emissions plot
figloce = os.path.join(figloc,'Emissions')
Gg = 1e-6 # conversion from kg to Gg
mcfi,ch4i,d13ci,foh,fst,fsl,fme,fch4,r13e = unpack(x_noaa)
emcf_opt = em0_mcf + mcf_shiftx(x_noaa)
ydrop = styear-1951 # production years not included in the optimization
rapids,mediums,slows,stocks = rapid[ydrop:],medium[ydrop:],slow[ydrop:],stock[ydrop:] # selected years
rapido = (1 - fst-fsl-fme)*rapids # optmized production distribution
mediumo = mediums + fme*rapids
slowo = slows + fsl*rapids
stocko =  stocks + fst*rapids
drapid,dmedium,dslow,dstock = rapido-rapids,mediumo-mediums,slowo-slows,stocko-stocks

plt.figure(figsize=(10,10))
plt.title('The default MCF emissions (Mccullogh & Midgley 2001; Rigby et al. 2008)')
plt.xlabel('Year')
plt.ylabel('MCF prod/emi (Gg/yr)')
plt.plot(years,em0_mcf*Gg,'ro-',label='Emissions')
plt.plot(fyears,prod*Gg,'ko-',label='Production')
plt.plot(fyears,rapid*Gg,'o-',color='c',label='Rapid production')
plt.plot(fyears,medium*Gg,'o-',color='b',label='Medium production')
plt.plot(fyears,slow*Gg,'o-',color='maroon',label='Slow production')
plt.plot(fyears,stock*Gg,'o-',color='steelblue',label='Stockpiling')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(figloce+'\\Base_mcf_emi.png')

plt.figure(figsize=(10,10))
plt.title('The relative contribution of each MCF production category over time')
plt.xlabel('Year')
plt.ylabel('Contribution (%)')
plt.plot(fyears,rapid/prod,'co-',label='Rapid')
plt.plot(fyears,medium/prod,'bo-',label='Medium')
plt.plot(fyears,slow/prod,'o-',color='maroon',label='Slow')
plt.plot(fyears,stock/prod,'o-',color='steelblue',label='Stock')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(figloce+'\\Base_contrib_per_category.png')

fig = plt.figure(figsize=(10,8))
ax1=fig.add_subplot(211)
ax2=fig.add_subplot(212)
ax1.set_title('Change in each production category after optimization\n\n Absolute')
ax2.set_title('Relative to prior')
ax1.set_ylabel('Difference (Gg/yr)')
ax2.set_ylabel('Difference (%)')
ax2.set_xlabel('Years')
ax1.plot(years,drapid*Gg,'o-',color='c',label='rapid')
ax1.plot(years,dmedium*Gg,'o-',color='b',label='medium')
ax1.plot(years,dslow*Gg,'o-',color='maroon',label='slow')
ax1.plot(years,dstock*Gg,'o-',color='k',label='stock')
ax2.plot(years,100*drapid/rapids,'o-',color='c')
ax2.plot(years,100*dmedium/mediums,'o-',color='b')
ax2.plot(years,100*dslow/slows,'o-',color='maroon')
ax2.plot(years,100*dstock/stocks,'o-',color='k')
#ax1.legend(loc='best')
plt.tight_layout()
plt.savefig(figloce+'\\Opt_contrib_per_category.png')

fig=plt.figure(figsize=(9,7))
ax1=fig.add_subplot(211)
ax2=fig.add_subplot(212)
#ax1.set_title('The difference between optimized and prior emissions\n\n Absolute')
#ax2.set_title('Relative to prior')
ax1.set_ylabel('Difference (Gg/yr)')
ax2.set_ylabel('Difference (%)')
ax2.set_xlabel('Years')
ax1.grid()
ax2.grid()
ax1.plot(years,(emcf_opt-em0_mcf)*Gg, 'o-', color='maroon',markersize=9,linewidth=1.8)
ax2.plot(years,100*(emcf_opt-em0_mcf)/em0_mcf, 'o-', color='maroon',markersize=9,linewidth=1.8)
plt.tight_layout()
plt.savefig(figloce+'\\Emi_diff_opt_base.png')


# Comparing my results to Rigby and Montzka
_,_,_,foh_noaa,_,_,_,_,_ = unpack(x_noaa)
_,_,_,foh_agag,_,_,_,_,_ = unpack(x_agag)
oh_rigby = read_glob_mean('OH_global_concentration_Rigby_2013.txt', 1992,2010, errors=True)
yr_rigby = [np.mean(oh_rigby[0][12*i:12*(i+1)]) for i in range(18)]
oh_rigbym = [np.mean(oh_rigby[1][12*i:12*(i+1)]) for i in range(18)]


fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(211)
#ax1.plot(oh_rigby[0],100*(oh_rigby[1]-max(oh_rigby[1]))/max(oh_rigby[1]), 'r--', label='AGAGE from Rigby (2013)')
ax1.plot(yr_rigby,100*(oh_rigbym-np.mean(oh_rigbym))/np.mean(oh_rigbym), 'r-', label='From Rigby (2013)',linewidth=4.0)
ax1.plot(years, 100*(foh_agag-np.mean(foh_agag))/np.mean(foh_agag), 'b-', label='Derived by me',linewidth=4.0)
ax1.legend(); 
ax1.grid()
ax1.set_ylabel('OH anomaly (%)')
ax2 = fig.add_subplot(212)
#ax1.plot(oh_rigby[0],100*(oh_rigby[1]-max(oh_rigby[1]))/max(oh_rigby[1]), 'r--', label='AGAGE from Rigby (2013)')
ax2.plot(yr_rigby,1000+(oh_rigbym-max(oh_rigbym))/max(oh_rigbym), 'r-', label='From Montzka (2011)',linewidth=6.0)
ax2.plot(years, 100*(foh_noaa-np.mean(foh_noaa))/np.mean(foh_noaa), 'b-', label='Derived by me',linewidth=6.0)
ax2.set_ylim([-6,8])
ax2.set_ylabel('OH anomaly (%)')
ax2.legend(loc='best')
ax2.grid()
plt.tight_layout()
plt.savefig(figloc+'\\OH_literature.png')

# CH4 mass balance
ch4_noaa,_,_ = forward_ch4(x_noaa)
loh_noaa = foh_noaa * l_ch4_oh * ch4_noaa
ech4_noaa = fch4_noaa * em0_ch4 / conv_ch4
lother_noaa = l_ch4_other * ch4_noaa
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_ylabel('Emissions/loss (ppb/yr)')
ax2.set_ylabel('CH4 growth (ppb/yr)')
ax1.plot(years,ech4_noaa,'go-',label='Emi NOAA')
ax1.plot(years,loh_noaa,'bo-', label='OH loss NOAA')
ax1.plot(years,lother_noaa,'ro-',label='Other loss NOAA')
ax2.plot(array(years[1:])-.5,ch4_growth_noaa,'o-',color='steelblue',label='CH4 growth data NOAA')
ax2.plot(array(years[1:])-.5,ch4_noaa[1:]-ch4_noaa[:-1],'o--',color='steelblue',label='CH4 growth model NOAA')


# d13C in emissions
_,_,_,_,_,_,_,fch4_noaa,r13e_noaa = unpack(x_noaa)
_,_,_,_,_,_,_,fch4_agag,r13e_agag = unpack(x_agag)
d13e_noaa = r13_to_d13c(r13e_noaa)
d13e_agag = r13_to_d13c(r13e_agag)
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(111)
ax1.set_ylabel(d13c_lab)
ax1.plot(years,d13e_noaa,'o-',color='steelblue',label='NOAA')
ax1.plot(years,d13e_agag,'o-',color='maroon',label='AGAGE')
ax1.legend(loc='best')
plt.tight_layout()
plt.savefig(figloc+'d13ce_noaa_agage.png')

bmb = 45e9 # BMB emissions
ech4 = fch4_agag*em0_ch4
sig_bmb = -22
sig_bio = -62
sig_fof = -44
E_bio = (ech4*(d13e_noaa-sig_fof) + bmb*(sig_fof-sig_bmb)) / (sig_bio-sig_fof)
E_fof = (ech4*(d13e_noaa-sig_bio) + bmb*(sig_bio-sig_bmb)) / (sig_fof-sig_bio)
print 'bmb',min(bmb/ech4),max(bmb/ech4)
print 'E_bio:',min(E_bio/ech4),max(E_bio/ech4)
print 'E_fof:',min(E_fof/ech4),max(E_fof/ech4)

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax3 = ax2.twinx()
ax1.set_ylabel(d13c_lab)
ax1.plot(years,d13e_noaa,'o-',color='steelblue',label='NOAA')
ax1.plot(years,d13e_agag,'o-',color='maroon',label='AGAGE')
ax2.set_ylabel('Biogenic (%)',color='green')
ax3.set_ylabel('Fossil fuel (%)',color='black')
ax2.plot(years,E_bio,'go-',label='Biogenic')
ax3.plot(years,E_fof,'ko-',label='Fossil fuel')
ax1.legend(loc='best')
ax2.legend(loc='lower center')
ax3.legend(loc='upper center')
plt.tight_layout()
plt.savefig(figloc+'\\CH4_source_comp.png')

x_noaa_ch = x_noaa[:]
r133 = d13c_to_r13([-53.5])
x_noaa_ch[-nt:] = array(r133*nt)
ch44,r13_ch4,_ = forward_ch4(x_noaa_ch)
d13cc = r13_to_d13c(r13_ch4)

plt.plot(d13cc,'ro')


# TEMPERATURE EFFECT ON REACTION CONSTANTS CH4 AND MCF
#Tt = np.linspace(272,274)
#kmcf0 = 1.64e-12*exp(-1520.0/temp)
#kmcf = 1.64e-12*exp(-1520.0/Tt)
#kch40 = 2.45e-12*exp(-1775.0/temp)
#kch4 = 2.45e-12*exp(-1775.0/Tt)
#kch4r = kch4/kch40-1
#ratio0 = kmcf/kch40
#ratio = kmcf/kch4
##plt.figure()
##plt.xlabel('Temperature')
##plt.ylabel('Change in kmcf/kch4 (%)')
##plt.title('The effect that a systematic bias in T will have on the ratio\n\
##             between the reaction constant with OH of MCF and CH4, relative\n\
##             to the T=272K basecase')
##plt.plot(Tt,100*(ratio/ratio0-1), 'r-')
##plt.grid()
##plt.tight_layout()
##plt.savefig('temp_effect_kch4_kmcf.png')
#plt.figure()
#plt.plot(Tt-temp,100*(kmcf/kmcf0-1), 'b-')
#plt.grid()
#plt.title('The effect that an increase in T over time will have on\n\
#            the MCF reaction rate, relative to the T=272K basecase')
#plt.xlabel('Temperature change')
#plt.ylabel('Change in kmcf (%)')
#plt.tight_layout()
#plt.savefig(figloc+'\\temp_effect_kmcf.png')