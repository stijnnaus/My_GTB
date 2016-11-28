# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:49:36 2016

@author: naus010
"""

from numpy import *
from math import *

nm = 12
errors = [1.,2.,3.,5.,8.,10.,15.,25.,60.]
vals = array([100.+20*(1-2*np.random.rand()) for i in range(nm)])
nruns = int(1e5)

# Varying monthly error
avss = []
for errori in errors:
    avs = []
    for i in range(nruns):
        vals_pert = np.random.normal(loc = vals, scale = errori, size = (nm))
        av = mean(vals_pert)
        avs.append(av)
    avss.append( array(avs) )
    
stds = [std(avss[i]) for i in range(len(avss))]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('The effect of changing the monthly error on\n \
                the standard deviation of the annual average')
ax1.set_xlabel('Monthly error (ppb)')
ax1.set_ylabel('Standard deviation (ppb)')
ax1.plot(errors, stds, 'ro')
ax1.plot(errors, array(errors) /sqrt(12), 'r--')
ax1.axis([0.,max(errors)+1,0,1.1*max(stds)])

# Varying nmonths
avss2 = []
nmonths = array([2,5,10,12,14,20,25,50])
errori = 1.
for nm in nmonths:
    avs = []
    valsi = array([100.+20*(1-2*np.random.rand()) for i in range(nm)])
    for i in range(nruns):
        vals_pert = np.random.normal(loc = valsi, scale = errori, size = (nm))
        av = mean(vals_pert)
        avs.append(av)
    avss2.append( array(avs) )

stds2 = [std(avss2[i]) for i in range(len(avss2))]

y = [ 1/sqrt(float(nmonths[i])) for i in range(len(nmonths))]
fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
ax1.set_title('The effect of changing the number of months\n \
                on the standard deviation of the annual average')
ax1.set_xlabel('Number of months')
ax1.set_ylabel('Standard deviation (ppb)')
ax1.plot(nmonths, stds2, 'ro')
ax1.plot(nmonths, y, 'r-')
ax1.axis([0.,max(nmonths)+1,0,1.1*max(stds2)])


# Non-constant monthly error

nm = 12
a = [1.,3.,5.,7.,10.,15.,30.]
errors = [ [a[j]*(1. + 2*np.random.rand()) for i in range(nm)] for j in range(len(a)) ]
vals = array([100.+20*(1-2*np.random.rand()) for i in range(nm)])

avss3 = []
for errorlist in errors:
    avs = []
    for i in range(nruns):
        vals_pert = np.random.normal(loc = vals, scale = errorlist, size = (nm) )
        av = mean(vals_pert)
        avs.append(av)
    avss3.append( array(avs) )
    
stds3 = [std(avss3[i]) for i in range(len(avss3))]

y = [sqrt(sum( array(errors[i])**2 )) / nm for i in range(len(a))]
fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
ax1.set_title('The effect of a non-constant monthly error\n \
                on the standard deviation of the annual average')
ax1.set_xlabel('Number of months')
ax1.set_ylabel('Standard deviation (ppb)')
ax1.plot(a, stds3, 'ro')
ax1.plot(a, y, 'r-')
ax1.axis( [0., max(a)+1, 0, 1.1*max(stds3)] )








































