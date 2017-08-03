#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:01:42 2017

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
#    ax.set_ylabel('Ratio')
    ax.set_ylabel('(High - Low) / Power Law')
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

### Read in the data from the .qdp files made by wdata in iplot on XSPEC
### pl eeuf po0
#dq, sets, res = qdp_read('xsseeuf_po0.qdp') ; c_set = ['g','r','b','m','c'] ; n_set = ['XMM','Suzaku','Swift','PIN','BAT']

### cutoffpl and adding in the zg
#dq1, sets, res = qdp_read('xssres_cutoffpl.qdp') ; c_set = ['g','r','b'] ; n_set = ['XMM','Suzaku','Swift']
#dq2, sets, res = qdp_read('xssres_cutoffpl+zg.qdp') ; c_set = ['g','r','b'] ; n_set = ['XMM','Suzaku','Swift']

#### extrapolated fit
#dq, sets, res = qdp_read('xssres_cutoffpl+zg_broadband.qdp') ; c_set = ['g','r','b','m','c'] ; n_set = ['XMM','Suzaku','Swift','PIN','BAT']

### simple broadband fits
#dq1, sets, res = qdp_read('xssres_cutoffpl+zg_broadbandFIT.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
#dq2, sets, res = qdp_read('xssres_cutoffpl+zg+bb_broadbandFIT.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']

### physical individual fits
#dq1, sets, res = qdp_read('xssres_distant_individualfits.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
#dq2, sets, res = qdp_read('xssres_blurred_individualfits.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']

### physical simultaneous fits
#dq1, sets, res = qdp_read('xssres_distant_simultaneousfits.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
#dq2, sets, res = qdp_read('xssres_blurred_simultaneousfits.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']

#### just the xmm and suz fits to show the iron line differences
#dq1, sets, res = qdp_read('xmm+suz_cutoffpl.qdp') ; c_set = ['g','r'] ; n_set = ['XMM','Suzaku']
#dq2, sets, res = qdp_read('xmm+suz_cutoffpl+zg.qdp') ; c_set = ['g','r'] ; n_set = ['XMM','Suzaku']

############
############

#### xmm, suz, and swift po0
#dq, sets, res = qdp_read('xmm+suz+swift_po0.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']

#### xmm, suz, and swift zg frozen and free fits
#dq1, sets, res = qdp_read('xmm+suz+swift_copl+zgN_Esigfrozen.qdp') ; c_set = ['g','r','b'] ; n_set = ['XMM','Suzaku','Swift']
#dq2, sets, res = qdp_read('xmm+suz+swift_copl+zgN_Esigfree.qdp') ; c_set = ['g','r','b'] ; n_set = ['XMM','Suzaku','Swift']

#### above fits extrapolated
#dq, sets, res = qdp_read('xmm+suz+swift_copl+zgN_Esigfree_broadband.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']

#### suz hilo fit
dq, sets, res = qdp_read('suz_hilo_res.qdp') ; c_set = ['r'] ; n_set = ['Suzaku']

#### broadband copl+zg fit
#dq1, sets, res = qdp_read('xmm+suz+swift_copl+zgN_Esigfree_broadband_fit.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
#dq2, sets, res = qdp_read('xmm+suz+swift_copl+zgN+bb_Esigfree_broadband_fit.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']

#### broadband physical fits
#dq1, sets, res = qdp_read('xmm+suz+swift_copl+reflionx_cold.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
#dq2, sets, res = qdp_read('xmm+suz+swift_copl+kblurXreflionx_blur.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']



### Plot up the data po0
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axhline(y=1.0, color='k', dashes=[5,3], linewidth=1.0)
for i in range(0,sets):
    x = i+1
    if (i == 0):
        shape = 'd'
    if (i == 1):
        shape = 'o'
    if (i == 2) and (sets < 4):
        shape = 's'
    if (i == 2) and (sets > 4):
        shape = 'o'
    if (i == 3):
        shape = 's'
    if (i == 4):
        shape = 's'
    plottertron(dq["e{0}".format(x)], dq["e_err{0}".format(x)], dq["f{0}".format(x)], dq["f_err{0}".format(x)], c_set[i], n_set[i], 0, 2, 'marker', shape)
#    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks([1,10]) # choose which x locations to have ticks
    ax.set_xticklabels([1,10]) # set the labels to display at those ticks
    ax.get_xaxis().get_minor_formatter().labelOnlyBase = True
    ax.set_xlim(0.7,10.0)
#    ax.set_ylim(0.5,2.5)
    ax.set_ylim(0.3,1.7)
    ax.tick_params(axis='both', which='both', direction='in', top='on', right='on')

#plt.savefig('/Users/agonzalez/Dropbox/Graduate/PhD/IIIZw2Paper2017/suzaku_hilo.png',bbox_inches='tight',dpi=300)


###### Gridspec plotting
#if (res == 0):
#    plt.figure()
#    gs1 = gridspec.GridSpec(2,1)
#    gs1.update(wspace=0, hspace=0)
#    for i in range(0,sets):
#        x = i+1
#
#        if (i == 0):
#            shape = 'd'
#        if (i == 1):
#            shape = 'o'
#        if (i == 2) and (sets < 4):
#            shape = 's'
#        if (i == 2) and (sets > 4):
#            shape = 'o'
#        if (i == 3):
#            shape = 's'
#        if (i == 4):
#            shape = 's'
#
#        ax = plt.subplot(gs1[0])
#        plottertron(dq1["e{0}".format(x)], dq1["e_err{0}".format(x)], dq1["f{0}".format(x)], dq1["f_err{0}".format(x)], c_set[i], n_set[i], 0, 3, 'marker', shape)
##        ax.set_xscale('log')
##        ax.fill_between(x=dq1["e{0}".format(x)], y1=dq1["f{0}".format(x)]-dq1["f_err{0}".format(x)], y2=dq1["f{0}".format(x)]+dq1["f_err{0}".format(x)], color=c_set[i], label=n_set[i], alpha=0.75)
##        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#        ax.set_xticks([2,5,10]) # choose which x locations to have ticks
#        ax.set_xticklabels([2,5,10]) # set the labels to display at those ticks
#        ax.get_xaxis().get_minor_formatter().labelOnlyBase = True
#        ax.set_xlim(0.3,150.)
#        ax.set_ylim(0.4,1.7)
##        ax.set_xlim(2.,10.)
##        ax.set_ylim(0.6,1.4)
#
#        ax = plt.subplot(gs1[1])
#        plottertron(dq2["e{0}".format(x)], dq2["e_err{0}".format(x)], dq2["f{0}".format(x)], dq2["f_err{0}".format(x)], c_set[i], n_set[i], 0, 3, 'marker', shape)
##        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#        ax.set_xticks([1,10,100]) # choose which x locations to have ticks
#        ax.set_xticklabels([1,10,100]) # set the labels to display at those ticks
#        ax.get_xaxis().get_minor_formatter().labelOnlyBase = True
#        ax.set_xlim(0.3,150.)
#        ax.set_ylim(0.4,1.7)
##        ax.set_xlim(2.,10.)
##        ax.set_ylim(0.6,1.4)
#
#    for i in range(0,2):
#        ax = plt.subplot(gs1[i])
#        if (i == 0):
#            ax.text(.25,.1,'cold distant',horizontalalignment='center',transform=ax.transAxes)
#        elif (i == 1):
#            ax.text(.25,.1,'blurred ionised',horizontalalignment='center',transform=ax.transAxes)
#
#        ax.tick_params(axis='both', which='both', direction='in', top='on', right='on')
#        if (i != 1):
#            ax.set_xticklabels([])
#        ax.axhline(y=1.0, color='k', dashes=[5,3], linewidth=1.0)
#        ax.axvline(x=6.4/(1.+0.089338), color='k', dashes=[5,3], linewidth=1.0)
#
#
##plt.savefig('/Users/agonzalez/Dropbox/Graduate/PhD/IIIZw2Paper2017/physical_broadband_fits.png', bbox_inches='tight', dpi=300)
