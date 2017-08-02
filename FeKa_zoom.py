#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:17:38 2017

@author: agonzalez
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
import math
import os
from matplotlib.ticker import ScalarFormatter
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally
plt.rc('font',family='serif')
plt.rc('text',usetex=True)


####################################################################################################
# read in the qdp data files
def qdp_read(filename):
    data = np.genfromtxt(filename, skip_header=3)
    
    # create an array that lists all rows with NO in them
    new_set = []
    new_set.append(0)
    for i in range (0, len(data)):
        if (math.isnan(data[i,0]) == 1):
            new_set.append(i)
    new_set.append(len(data))
    
    # determine the number of data sets being read in 
    n = len(new_set)-1
    r = 0
    
    # create a dictionary that stores each column according to its data set
    d = {}
    for i in range(0,n):
        x = i+1
        if (i == 0):
            new_set[i] = -1
        if (i == len(new_set)-2):
            new_set[i+1] = new_set[i+1]+1
        d["e{0}".format(x)]     =data[new_set[i]+1:new_set[i+1]-1,0]
        d["e_err{0}".format(x)] =data[new_set[i]+1:new_set[i+1]-1,1]
        d["f{0}".format(x)]     =data[new_set[i]+1:new_set[i+1]-1,2]
        d["f_err{0}".format(x)] =data[new_set[i]+1:new_set[i+1]-1,3]

    return d, n, r
####################################################################################################


####################################################################################################
# Plot x and y with labels (trying to do it)
def plottertron(x, x_error, y, y_error, line_color, line_label, legyn, legloc, style, mrkshp):
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Ratio')
    ax.set_xscale('log')
    
#    ax.set_yscale('log')
#    ax.set_ylabel('Normalized Counts (s$^{-1}$ keV$^{-1}$)')
#    ax.set_ylabel('keV$^{2}$ (Photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$)')   # use this when plotting eeuf 
    if (style == 'line'):
        ax.errorbar(x, y, xerr=x_error, yerr=y_error, ecolor=line_color, capthick=1, color=line_color, label=line_label)
    elif (style == 'marker'):
        ax.errorbar(x, y, xerr=x_error, yerr=y_error, fmt=mrkshp, markersize=5, ecolor=line_color, capthick=1, linewidth = 1, color=line_color, label=line_label, markerfacecolor='none', alpha=1.0)
    plt.show()
    if (legyn == 1):
        plt.legend(loc=legloc, ncol=2, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)
#        plt.legend(loc=legloc, labelspacing=0.1, fontsize=16)
        ax.get_legend()
####################################################################################################

os.chdir("/Users/agonzalez/Documents/Research/Data/Mrk1501/newswift/rebinned_July26")

#### xmm, suz, and swift po0        
#dq, sets, res = qdp_read('xmm+suz+swift_po0.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
        
#### xmm, suz, and swift zg frozen and free fits
#dq1, sets, res = qdp_read('xmm+suz+swift_copl+zgN_Esigfrozen.qdp') ; c_set = ['g','r','b'] ; n_set = ['XMM','Suzaku','Swift']
dq1, sets, res = qdp_read('xmm+suz+swift_copl_Fezoom.qdp') ; c_set = ['g','r','b'] ; n_set = ['XMM','Suzaku','Swift']
dq2, sets, res = qdp_read('xmm+suz+swift_copl+zgN_Esigfree.qdp') ; c_set = ['g','r','b'] ; n_set = ['XMM','Suzaku','Swift']

#### above fits extrapolated
#dq, sets, res = qdp_read('xmm+suz+swift_copl+zgN_Esigfree_broadband.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']

#### suz hilo fit
#dq, sets, res = qdp_read('suz_hilo_res.qdp') ; c_set = ['r'] ; n_set = ['Suzaku']

#### broadband copl+zg fit
#dq1, sets, res = qdp_read('xmm+suz+swift_copl+zgN_Esigfree_broadband_fit.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
#dq2, sets, res = qdp_read('xmm+suz+swift_copl+zgN+bb_Esigfree_broadband_fit.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']

#### broadband physical fits
#dq1, sets, res = qdp_read('xmm+suz+swift_copl+reflionx_cold.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
#dq2, sets, res = qdp_read('xmm+suz+swift_copl+kblurXreflionx_blur.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']


##### Gridspec plotting 
plt.figure()
gs = gridspec.GridSpec(3,1)
gs.update(wspace=0, hspace=0)

ax1 = plt.subplot(gs[0]); ax1.set_xlabel('Energy (keV)'); ax1.set_ylabel('Ratio'); ax1.set_xscale('log')
ax2 = plt.subplot(gs[1]); ax2.set_xlabel('Energy (keV)'); ax2.set_ylabel('Ratio'); ax2.set_xscale('log')
ax3 = plt.subplot(gs[2]); ax3.set_xlabel('Energy (keV)'); ax3.set_ylabel('Ratio'); ax3.set_xscale('log')

x=1
ax1.axhline(y=1.0, color='k', dashes=[5,3], linewidth=1)
ax1.axvline(x=6.4/(1.+0.089338), color='k', dashes=[5,3], linewidth=1)
ax1.errorbar(x=dq1["e{0}".format(x)], y=dq1["f{0}".format(x)], xerr=dq1["e_err{0}".format(x)], yerr=dq1["f_err{0}".format(x)], fmt='d', markersize=5, markerfacecolor='none', ecolor='g', capthick=0, color='g', linewidth=1, alpha=1.0)
#ax1.errorbar(x=dq2["e{0}".format(x)], y=dq2["f{0}".format(x)], xerr=dq2["e_err{0}".format(x)], yerr=dq2["f_err{0}".format(x)], fmt='d', markersize=5, markerfacecolor='g', ecolor='g', capthick=0, color='g', linewidth=1, alpha=1.0)
#ax1.fill_between(x=dq1["e{0}".format(x)], y1=dq1["f{0}".format(x)]-dq1["f_err{0}".format(x)], y2=dq1["f{0}".format(x)]+dq1["f_err{0}".format(x)], facecolor='none', edgecolor='g', hatch='+', linewidth=1.0, alpha=1.0)
ax1.fill_between(x=dq2["e{0}".format(x)], y1=dq2["f{0}".format(x)]-dq2["f_err{0}".format(x)], y2=dq2["f{0}".format(x)]+dq2["f_err{0}".format(x)], facecolor='g', edgecolor='g', linewidth=1.0, alpha=0.5)
ax1.set_xticks([5,6]) # choose which x locations to have ticks
ax1.set_xticklabels([5,6]) # set the labels to display at those ticks
ax1.get_xaxis().get_minor_formatter().labelOnlyBase = True
ax1.set_xlim(4.,8.)
ax1.set_ylim(0.8,1.35)
ax1.tick_params(axis='both', which='both', direction='in', top='on', right='on')

x=2
ax2.axhline(y=1.0, color='k', dashes=[5,3], linewidth=1)
ax2.axvline(x=6.4/(1.+0.089338), color='k', dashes=[5,3], linewidth=1)
ax2.errorbar(x=dq1["e{0}".format(x)], y=dq1["f{0}".format(x)], xerr=dq1["e_err{0}".format(x)], yerr=dq1["f_err{0}".format(x)], fmt='o', markersize=5, markerfacecolor='none', ecolor='r', capthick=0, color='r', linewidth=1, alpha=1.0)
#ax2.errorbar(x=dq2["e{0}".format(x)], y=dq2["f{0}".format(x)], xerr=dq2["e_err{0}".format(x)], yerr=dq2["f_err{0}".format(x)], fmt='o', markersize=5, markerfacecolor='r', ecolor='r', capthick=0, color='r', linewidth=1, alpha=1.0)
#ax2.fill_between(x=dq1["e{0}".format(x)], y1=dq1["f{0}".format(x)]-dq1["f_err{0}".format(x)], y2=dq1["f{0}".format(x)]+dq1["f_err{0}".format(x)], facecolor='none', edgecolor='r', hatch='+', linewidth=1.0, alpha=1.0)
ax2.fill_between(x=dq2["e{0}".format(x)], y1=dq2["f{0}".format(x)]-dq2["f_err{0}".format(x)], y2=dq2["f{0}".format(x)]+dq2["f_err{0}".format(x)], facecolor='r', edgecolor='r', linewidth=1.0, alpha=0.5)
ax2.set_xticks([5,6]) # choose which x locations to have ticks
ax2.set_xticklabels([5,6]) # set the labels to display at those ticks
ax2.get_xaxis().get_minor_formatter().labelOnlyBase = True
ax2.set_xlim(4.,8.)
ax2.set_ylim(0.8,1.35)
ax2.tick_params(axis='both', which='both', direction='in', top='on', right='on')

x=3
ax3.axhline(y=1.0, color='k', dashes=[5,3], linewidth=1)
ax3.axvline(x=6.4/(1.+0.089338), color='k', dashes=[5,3], linewidth=1)
ax3.errorbar(x=dq1["e{0}".format(x)], y=dq1["f{0}".format(x)], xerr=dq1["e_err{0}".format(x)], yerr=dq1["f_err{0}".format(x)], fmt='s', markersize=5, markerfacecolor='none', ecolor='b', capthick=0, color='b', linewidth=1, alpha=1.0)
#ax3.errorbar(x=dq2["e{0}".format(x)], y=dq2["f{0}".format(x)], xerr=dq2["e_err{0}".format(x)], yerr=dq2["f_err{0}".format(x)], fmt='s', markersize=5, markerfacecolor='b', ecolor='b', capthick=0, color='b', linewidth=1, alpha=1.0)
#ax3.fill_between(x=dq1["e{0}".format(x)], y1=dq1["f{0}".format(x)]-dq1["f_err{0}".format(x)], y2=dq1["f{0}".format(x)]+dq1["f_err{0}".format(x)], facecolor='none', edgecolor='b', hatch='+', linewidth=1.0, alpha=1.0)
ax3.fill_between(x=dq2["e{0}".format(x)], y1=dq2["f{0}".format(x)]-dq2["f_err{0}".format(x)], y2=dq2["f{0}".format(x)]+dq2["f_err{0}".format(x)], facecolor='b', edgecolor='b', linewidth=1.0, alpha=0.5)
ax3.set_xticks([4,5,6,7,8]) # choose which x locations to have ticks
ax3.set_xticklabels([4,5,6,7,8]) # set the labels to display at those ticks
ax3.get_xaxis().get_minor_formatter().labelOnlyBase = True
ax3.set_xlim(4.,8.)
ax3.set_ylim(0.8,1.8)
ax3.tick_params(axis='both', which='both', direction='in', top='on', right='on')


#plt.savefig('/Users/agonzalez/Dropbox/Graduate/PhD/IIIZw2Paper2017/individual_Fezoom.png', bbox_inches='tight', dpi=300)
#plt.savefig('/Users/agonzalez/Desktop/individual_Fezoom_semifilled.png', bbox_inches='tight', dpi=300)