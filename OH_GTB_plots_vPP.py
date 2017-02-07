# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:30:54 2017

@author: naus010

The file in which I can combine experiments, saved from OH_GTB.
I should run OH_GTB_read_data_helper, then OH_GBT, then this file

vPP: This is the setup that produces powerpoint images. Mostly larger text.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams

rcParams.update({'font.size': 16})
rcParams.update({'lines.markersize': 9})
rcParams.update({'lines.linewidth': 1.5})

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
    ax1 = fig.add_subplot(411); plt.locator_params(axis='y',nbins=6) 
    ax2 = fig.add_subplot(412); plt.locator_params(axis='y',nbins=6)
    ax3 = fig.add_subplot(413); plt.locator_params(axis='y',nbins=6)
    ax4 = fig.add_subplot(414); plt.locator_params(axis='y',nbins=6)
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
    lgd=ax2.legend(bbox_to_anchor=(1.45,1.),title=legtit)
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
    fig = plt.figure(figsize=(15,50))
    ax1 = fig.add_subplot(211) # MCF observations
    ax2 = fig.add_subplot(212) # CH4 observations
    ax1.set_title(plottit+'\n\nMethyl chloroform')
    ax2.set_title(r'Methane')
    ax1.set_ylabel('MCF (ppt)'); ax2.set_ylabel(r'CH$_4$ (ppb)')
    ax1.grid(True); ax2.grid(True)
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
    plt.locator_params(axis='y',nbins=5)
    lgd=ax2.legend(bbox_to_anchor=(1.32,1.),title=legtit)
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
    plt.locator_params(axis='y',nbins=5)
    ax2 = fig.add_subplot(312) # CH4 observations
    plt.locator_params(axis='y',nbins=5)
    ax3 = fig.add_subplot(313) # d13C observations
    plt.locator_params(axis='y',nbins=5)
    #ax1.set_title(plottit+'\n\nMethyl chloroform')
    #ax2.set_title(r'Methane')
    #ax3.set_title(r'$\delta^{13}$C in CH$_4$')
    ax1.set_ylabel('MCF (ppt)'); ax2.set_ylabel(r'CH$_4$ (ppb)')
    ax3.set_ylabel(r'$\delta^{13}$C (permil)')
    #ax3.set_xlabel('Year')
    ax1.grid(True); ax2.grid(True); ax3.grid(True)
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
    lgd=ax2.legend(bbox_to_anchor=(1.32,1.),title=legtit)
    fig.tight_layout()
    if figname==None: figname='default'
    plt.savefig(figloc+'\\'+figname,bbox_extra_artists=(lgd,), bbox_inches='tight')
    
def fig_oh_v_ch4growth(xopts,labels,plottit,legtit=None,dataset='noaa',figname=None):
    '''
    Plots the CH4 growth rate versus OH concentrations.
    '''
    if dataset=='noaa':
        ch4_growth = ch4_noaa[1:] - ch4_noaa[:-1]
    elif dataset=='agage':
        ch4_growth = ch4_agage[1:] - ch4_agage[:-1]
    else: print 'Select a valid dataset: NOAA or AGAGE!'
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.grid(True)
    plt.locator_params(axis='y',nbins=6)
    #ax1.set_title(plottit+'\n\n The correlation between the OH and the CH4 growth rate')
    ax1.set_xlabel('CH4 growth rate (ppb/yr)')
    ax1.set_ylabel(oh_lab)
    for i,xopt in enumerate(xopts):
        _,_,_,fohi,_,_,_,_,_ = unpack(xopt)
        ohu = fohi*oh
        oh_mid = array([(ohu[j]+ohu[j-1])/2 for j in range(1,nt)])
        cor = round(np.corrcoef(ch4_growth,oh_mid)[0,1],2)
        ax1.plot(ch4_growth, oh_mid/1e6, 'o',color=dif_col[i+1], label=labels[i]+'(r='+str(cor)+')')
    ax1.set_xlim([min(ch4_growth)-1,max(ch4_growth)+1])
    ax1.set_ylim([0.86,0.91])
    #lgd=ax1.legend(bbox_to_anchor=(1.35,1.),title=legtit)
    fig.tight_layout()
    if figname==None: figname='default'
    plt.savefig(figloc+'\\'+figname,bbox_extra_artists=(lgd,), bbox_inches='tight')
    
def fig_lifetime(xopts,labels,plottit,legtit=None ,figname=None):
    '''
    Plot of MCF and CH4 lifetimes against OH and the total lifetime.
    '''
    fig = plt.figure()
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
        ax1.plot(yrs, tau_mcf_oh, 'o', label='vs oh '+labels[i])
        ax1.plot(yrs, tau_mcf_tot, 'v', label='tot '+labels[i])
        ax2.plot(yrs, tau_ch4_oh, 'o', label='vs oh '+labels[i])
        ax2.plot(yrs, tau_ch4_tot, 'v', label='tot '+labels[i])
    lgd=ax1.legend(bbox_to_anchor=(1.4,1.),title=legtit)
    fig.tight_layout()
    if figname==None: figname='default'
    plt.savefig(figloc+'\\'+figname,bbox_extra_artists=(lgd,), bbox_inches='tight')
        
def cor_calc(x_opt,dataset='noaa'):
    '''
    Calculates the correlation coefficients between several optimized and/or
    observed parameters.
    Returns correlation between: CH4 growth and OH; CH4 growth and CH4 emissions;
                CH4 emissions and OH; MCF emissions and OH
    '''
    if dataset=='noaa':
        ch4_growth = [ch4_noaa[i]-ch4_noaa[i-1] for i in range(1,nt)]
    elif dataset=='agage':
        ch4_growth = [ch4_agage[i]-ch4_agage[i-1] for i in range(1,nt)]
    
    _,_,_,fohi,fsti,fsli,fmei,fch4i,_ = unpack(x_agag)
    ohu = fohi*oh
    emcfu = em0_mcf+mcf_shift(fsti,fsli,fmei)
    ech4u = fch4i*em0_ch4
    oh_mid = array([(ohu[j]+ohu[j-1])/2 for j in range(1,nt)])
    emcfu_mid = array([(emcfu[j]+emcfu[j-1])/2 for j in range(1,nt)])
    ech4_mid = array([(ech4u[j]+ech4u[j-1])/2 for j in range(1,nt)])
    cor_groh = round(np.corrcoef(ch4_growth,oh_mid)[0,1],2)
    cor_grec = round(np.corrcoef(ch4_growth,ech4_mid)[0,1],2)
    cor_ecoh = round(np.corrcoef(oh_mid,ech4_mid)[0,1],2)
    cor_emoh = round(np.corrcoef(oh_mid,emcf_mid)[0,1],2)
    return cor_groh, cor_grec, cor_ecoh, cor_emoh
        
dif_col = ['steelblue','maroon','blue','pink','black','cyan'] # Very different colors
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
x_normal = read_x('normal2', 'noaa')
x_noch4 = read_x('noch4', 'noaa')
x_nomcf = read_x('offmcf', 'noaa')
x_nooh = read_x('offoh', 'noaa')
x_onoff = [x_normal, x_noch4, x_nomcf, x_nooh]
lab_onoff = ['normal','no ch4','no mcf','no oh'] # label per experiment
title_onoff = 'The effect of excluding one of the parameters from the optimization' # Plot title
name_onoff = 'onoff_reldev.png' # Figure name
name_onoff2 = 'onoff_obs.png'
# Drawing&saving the figure:
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

# d13C not optimized
x_no13c = read_x('nod13c','noaa')
x_no13coh = read_x('nod13coh','noaa')
lab_no13c = ['no d13C', 'also no OH']
name_no13c = 'nod13C_reldev.png'
name_no13c2 = 'nod13C_obs.png'
leg_no13c = ''
tit_no13c = 'The effect of optimizing all but d13C'
fig_reldev([x_no13c,x_no13coh],lab_no13c,tit_no13c,legtit=leg_no13c,figname=name_no13c)
fig_all_obs([x_no13c,x_no13coh], lab_no13c, tit_no13c, dataset='noaa',figname=name_no13c2)


# Fitting OH to CH4 or MCF
x_ohtomcf = read_x('nomcfem','noaa')
x_ohtoch4 = read_x('noch4em','noaa')
fig_reldev([x_noaa,x_ohtomcf,x_ohtoch4], ['Normal','Match MCF',r'Match CH$_4$'],'',figname='OH_tomatch_MCF_CH4.png')


# Default normal obs plot
fig_all_obs([x_noaa],['NOAA'],'',dataset='noaa',figname='standard_obs_plot.png')
fig_reldev([x_noaa],['NOAA'],'',figname='standard_reldev')

# Correlation plots
fig_oh_v_ch4growth([x_noaa,x_noch4],['',''],'')
fig_oh_v_ch4growth([x_noaa],['NOAA'],'',figname='standard_growth_cor')
fig_oh_v_ch4growth([x_noch4],[''],'',figname='noch4_growth_cor')

# lifetime plots
fig_lifetime([x_noaa],[' '],'',figname='lifetime_noaa')

_,_,_,fohi,_,_,_,fch4i,_ = unpack(x_agag)
ohu = fohi*oh
oh_mid = array([(ohu[j]+ohu[j-1])/2 for j in range(1,nt)])
ech4u = fch4i*em0_ch4
ech4_mid = array([(ech4u[j]+ech4u[j-1])/2 for j in range(1,nt)])
ch4_growth = [ch4_agage[i]-ch4_agage[i-1] for i in range(1,nt)]
cor = round(np.corrcoef(ch4_growth,oh_mid)[0,1],2)
cor2 = round(np.corrcoef(ch4_growth,ech4_mid)[0,1],2)
cor3 = round(np.corrcoef(oh_mid,ech4_mid)[0,1],2)


_,_,_,fohi,_,_,_,fch4i,_ = unpack(x_noch4)
ohu = fohi*oh
oh_mid = array([(ohu[j]+ohu[j-1])/2 for j in range(1,nt)])
ch4_growth = [ch4_noaa[i]-ch4_noaa[i-1] for i in range(1,nt)]
cor = round(np.corrcoef(ch4_growth,oh_mid)[0,1],2)









dif = (mcf_noaa-mcf_agage)/mcf_obs_e
dif = (ch4_noaa-ch4_agage)/ch4_obs_e



