# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:47:56 2016

@author: naus010
"""

import numpy as np
import matplotlib.pyplot as plt
import math as M
from scipy.stats import norm
import random
import scipy.sparse as scs
import time
from mpl_toolkits.mplot3d import Axes3D
import scipy as sc

# Universal constants
R_ST = 11237.2e-6 # Standard V-PDP 13C/12C ratio
m = 2.767*10**18
xmair = 28.5
xmcf = 133.5
xch4 = 16.0
l_ocean = 83.0*365.0  # lifetime in days
l_strat = 45.0*365.0
oh = 0.90e6  # molecules/cm3
temp = np.array([272.0,287.0])  # Kelvin        
l_oh_mcf = 1.64e-12*np.exp(-1520.0/temp)*oh  # in seconds
l_oh_ch4 = 2.45e-12*np.exp(-1775.0/temp)*oh  # in seconds
l_oh_mcf /= 3600.*24.0  # in days
l_oh_ch4 /= 3600.*24.0  # in days
nyear = 21

plt.plot(temp,l_oh_mcf)
plt.plot(temp,l_oh_ch4)

print l_oh_mcf[0]/l_oh_ch4[0], l_oh_mcf[1]/l_oh_ch4[1]

def forward_mcf():
    
    
    
def forward_12ch4():
    
def forward_13ch4():

def adjoint_mcf():
    
def adjoint_12ch4():
    
def adjoint_13ch4():
    
#def precon_to_state():
#    
#def state_to_precon():
#    

def deltot_to_split(delc, totCH4):
    '''
    Converts a delta 13C value and a total CH4 quantity 
    (emission/concentration) to 12CH4 and 13CH4.
    '''
    q = R_ST * ( delc + 1 )
    12CH4 = totCH4 / (1+q)
    13CH4 = q * 12CH4
    return 12CH4, 13CH4
    
def split_to_deltot(12CH4, 13CH4):
    '''
    Converts a split 12CH4 and 13CH4 quantitities (emission/concentration)  
    to a delta 13C value and a total CH4 quantity.
    '''
    R_sample = 13CH4 / 12CH4
    delCH4 = ( R_sample / R_ST ) - 1
    totCH4 = 12CH4 + 13CH4
    return delCH4, totCH4
    
