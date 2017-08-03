#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:36:32 2017

@author: agonzalez
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import os

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally
plt.rc('font',family='serif')
plt.rc('text',usetex=True)


####################################################################################################
def error_avg(cts, err_cts, avg):
    ssq, s = 0., 0.
    n = len(cts)
    for i in range (0,n):
        ssq += (cts[i]-avg)**2.0
    ssq = ssq / (n-1.)
    s = np.sqrt(ssq/n)
    return s
####################################################################################################

####################################################################################################
def wmean(cts, err_cts):
    n = len(cts)
    w = 1./np.array(err_cts)
    top, bot, mean, err_mean = 0.,0.,0.,0.
    for i in range (0,n):
        top += cts[i]*w[i]**2.0
        bot += w[i]**2.0
    mean = top/bot
    err_mean = np.sqrt(1./bot)
    return mean, err_mean
####################################################################################################

####################################################################################################
def chisqcalc(dat, dat_err, avg):
    chisq, red_chisq = 0., 0.
    n = len(dat)
    for i in range (0,n):
        chisq += (dat[i]-avg)**2.0 / dat_err[i]**2.0
    red_chisq = chisq / (n-1.)
    return chisq, red_chisq
####################################################################################################

os.chdir("/Users/agonzalez/Documents/Research/Data/Mrk1501")

#d = np.genfromtxt('ratio_0.5-1.0_2.0-10.0.qdp', skip_header=3)
#d = np.genfromtxt('ratio_0.5-2.0_2.0-10.0.qdp', skip_header=3)     # old data set
d = np.genfromtxt('ratio_full_0.5-1_2-10.qdp', skip_header=3)      # new data set


# normalise the light curve so the time starts exactly at 0.0
d[:,0] = d[:,0] - d[0,0]

waL, e_waL = wmean(d[8:17,6], d[:17,7])
csL, rcsL = chisqcalc(d[8:17,6], d[:17,7], waL)
print 'LO wavg           = ', waL, ' +/-', e_waL
print 'LO red_chisq wavg = ', rcsL

waH, e_waH = wmean(d[22:,6], d[22:,7])
csH, rcsH = chisqcalc(d[22:,6], d[22:,7], waH)
print 'HI wavg           = ', waH, ' +/-', e_waH
print 'HI red_chisq wavg = ', rcsH

wa, e_wa = wmean(d[:,6], d[:,7])
cs, rcs = chisqcalc(d[:,6], d[:,7], wa)
print 'weighted avg                            = ', wa, ' +/-', e_wa
print 'red_chisq for fit to total weighted avg = ', rcs

# plot the hardness ratio
fig = plt.figure()
ax12 = plt.subplot(111)
ax12.errorbar(x=d[:,0], y=d[:,6], xerr=d[:,1], yerr=d[:,7], fmt='o', markersize=5, ecolor='r', capthick=1, linewidth=1, color='r', markerfacecolor='none', label='Hardness Ratio')
#ax12.fill_between(x=[d[0,0]-d[0,1],d[-1,0]+d[-1,1]], y1=wa-e_wa, y2=wa+e_wa, color='k', alpha=0.25)
ax12.axhline(y=wa, color='k', dashes=[5,3], linewidth=1)
#ax12.fill_between(x=d[22:,0], y1=waH-e_waH, y2=waH+e_waH, color='g', alpha=0.50)
#ax12.fill_between(x=d[8:17:,0], y1=waL-e_waL, y2=waL+e_waL, color='b', alpha=0.50)
ax12.axvline(d[17,0]-d[17,1], color='k', dashes=[5,3], linewidth=1)      # below is low state
ax12.axvline(d[22,0]-d[22,1], color='k', dashes=[5,3], linewidth=1)     # above is high state
ax12.text(.40,.75,'low\nflux',horizontalalignment='center',transform=ax12.transAxes)
ax12.text(.87,.75,'high\nflux',horizontalalignment='center',transform=ax12.transAxes)
ax12.set_xlim(-d[0,1],d[29,0]+d[29,1])
ax12.tick_params(axis='both', which='both', direction='in', top='on', right='on')
ax12.set_xlabel('Time (s)')
ax12.set_ylabel('Hardness Ratio')

#plt.savefig('/Users/agonzalez/Dropbox/Graduate/PhD/IIIZw2Paper2017/hardratio.png', bbox_inches='tight', dpi=300)




### Small subfigures of the lightcurves ####################################################################
#d[:,2], d[:,3], d[:,4], d[:,5], d[:,8], d[:,9] = d[:,2]*0.5094, d[:,3]*0.5094, d[:,4]*0.5094, d[:,5]*0.5094, d[:,8]*0.5094, d[:,9]*0.5094
#plt.figure()
#gs1 = gridspec.GridSpec(2,1)
#gs1.update(wspace=0, hspace=0)
#
#ax = plt.subplot(gs1[0])
#ax.errorbar(x=d[:,0], y=d[:,2], xerr=d[:,1], yerr=d[:,3], fmt='o', markersize=3, ecolor='r', capthick=1, linewidth=1, color='r', label='0.5 - 2 keV')
#ax.errorbar(x=d[:,0], y=d[:,4], xerr=d[:,1], yerr=d[:,5], fmt='o', markersize=3, ecolor='b', capthick=1, linewidth=1, color='b', label='2 - 10 keV')
##ax.errorbar(x=d[:,0], y=d[:,8], xerr=d[:,1], yerr=d[:,9], fmt='o', markersize=3, ecolor='k', capthick=1, linewidth=1, color='k', label='0.5 - 10')
#ax.set_xlim(-d[0,1],d[29,0]+d[29,1])
#ax.set_ylim(0.0,0.20) # lower bound 0.20
#ax.tick_params(axis='both', which='both', direction='in', top='on', right='on')
#ax.set_xticklabels([])
#ax.set_ylabel('Counts / sec')
#plt.legend(loc=4, ncol=1, labelspacing=0.1, fontsize=16)
#ax.get_legend()
##ax.set_aspect('equal')
#
#ax = plt.subplot(gs1[1])
#ax.errorbar(x=d[:,0], y=d[:,6], xerr=d[:,1], yerr=d[:,7], fmt='o', markersize=3, ecolor='k', capthick=1, linewidth=1, color='k', label='Hardness Ratio')
#ax.set_xlim(-d[0,1],d[29,0]+d[29,1])
#ax.tick_params(axis='both', which='both', direction='in', top='on', right='on')
#ax.set_ylabel('Ratio')
#ax.set_xlabel('Time (s)')
##ax.set_aspect('equal')
#
##plt.savefig('hardratio_0.5-2_2-10.png', bbox_inches='tight', dpi=300)
##plt.savefig('../../LaTeX/IIIZw2/2-10_res_subplot.png',bbox_inches='tight',dpi=300)
#
#
#
##
##fig = plt.figure()
###fig = plt.figure(figsize=(6,2.5))
##ax11 = plt.subplot(111)
##ax11.errorbar(x=d[:,0], y=d[:,2], xerr=d[:,1], yerr=d[:,3], fmt='o', markersize=3, ecolor='r', capthick=1, linewidth=1, color='r', label='0.5 - 2.0')
##ax11.errorbar(x=d[:,0], y=d[:,4], xerr=d[:,1], yerr=d[:,5], fmt='o', markersize=3, ecolor='b', capthick=1, linewidth=1, color='b', label='2.0 - 10.0')
##ax11.set_xlim(-d[0,1],d[29,0]+d[29,1])
##ax11.set_ylim(0.0,0.20) # lower bound 0.20
##ax11.tick_params(axis='both', which='both', direction='in', top='on', right='on')
##ax11.set_xlabel('Time (s)')
##ax11.set_ylabel('Counts / sec')
##plt.legend(loc=3, labelspacing=0.1, fontsize=16)
##ax11.get_legend()
