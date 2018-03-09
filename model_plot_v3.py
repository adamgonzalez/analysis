#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:47:47 2017

@author: agonzalez
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
import math
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

    # determine if the file being read in has residuals in it or not
    if (math.isnan(data[new_set[int(len(new_set)/2.)]+1,4]) == 1):
        n = int(len(new_set)/2.)
        r = 1
    elif (math.isnan(data[new_set[int(len(new_set)/2.)]+1,4]) != 1):
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
        d["m{0}".format(x)]     =data[new_set[i]+1:new_set[i+1]-1,4]

    for i in range(n,len(new_set)-1):
        x = i+1-int(len(new_set)/2.)
        if (i == len(new_set)-2):
            new_set[i+1] = new_set[i+1]+1
        d["res_e{0}".format(x)]     =data[new_set[i]+1:new_set[i+1]-1,0]
        d["res_e_err{0}".format(x)] =data[new_set[i]+1:new_set[i+1]-1,1]
        d["res{0}".format(x)]       =data[new_set[i]+1:new_set[i+1]-1,2]
        d["res_err{0}".format(x)]   =data[new_set[i]+1:new_set[i+1]-1,3]

    return d, n, r
####################################################################################################


####################################################################################################
# Plot x and y with labels (trying to do it)
def plottertron(x, x_error, y, y_error, line_color, line_label, xscale, yscale, legyn, legloc, style, mrkshp):
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Ratio')
    #ax.set_ylabel('(High - Low) Ratio')
    if (xscale == 'log'):
        ax.set_xscale('log')
    if (yscale == 'log'):
#        ax.set_ylabel('Normalized Counts (s$^{-1}$ keV$^{-1}$)')
        ax.set_ylabel('keV$^{2}$ (Photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$)')   # use this when plotting eeuf
        ax.set_yscale('log')
    if (style == 'line'):
        ax.errorbar(x, y, xerr=x_error, yerr=y_error, ecolor=line_color, capthick=1, color=line_color, label=line_label)
    elif (style == 'marker'):
        ax.errorbar(x, y, xerr=x_error, yerr=y_error, fmt=mrkshp, markersize=0, ecolor=line_color, capthick=1, linewidth = 1, color=line_color, label=line_label, markerfacecolor='none')
#        ax.fill_between(x=x, y1=y-y_error, y2=y+y_error, color=line_color, alpha=0.75)
#        ax.scatter(x,y, )
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#    ax.set_xticks([2,5,10]) # choose which x locations to have ticks
#    ax.set_xticklabels([2,5,10]) # set the labels to display at those ticks
#    ax.get_xaxis().get_minor_formatter().labelOnlyBase = True
    plt.show()
    if (legyn == 1):
        plt.legend(loc=legloc, ncol=2, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)
#        plt.legend(loc=legloc, labelspacing=0.1, fontsize=16)
        ax.get_legend()
####################################################################################################



### Read in the data from the .qdp files made by wdata in iplot on XSPEC
#dq, sets, res = qdp_read('powerlaw2-10extr.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']

#### OLD ####
### Unfolded
#dq, sets, res = qdp_read('po0.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
### 2-10 fits
#dq1, sets, res = qdp_read('copl.qdp') ; c_set = ['g','r','b'] ; n_set = ['XMM','Suzaku','Swift']
#dq2, sets, res = qdp_read('copl+zgN.qdp') ; c_set = ['g','r','b'] ; n_set = ['XMM','Suzaku','Swift']
#dq2, sets, res = qdp_read('copl+zgB.qdp') ; c_set = ['g','r','b'] ; n_set = ['XMM','Suzaku','Swift']
#dq, sets, res = qdp_read('copl+zgB_extr.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
#### Broadband fits
#dq1, sets, res = qdp_read('copl+zgB_broad_J1.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
#dq2, sets, res = qdp_read('copl+zgB+bb_broad_J1.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
### Reflection models
#dq, sets, res = qdp_read('distant_constant_reflection.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
#dq, sets, res = qdp_read('distant_constant_reflection.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
#dq, sets, res = qdp_read('distant_constant_reflection.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
### Sophia's Stuff
#dq, sets, res = qdp_read('Sophia/suzakupl2-10ext.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']


### NEW ####
#dq, sets, res = qdp_read('newswift/po0.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
### 2-10 fits
#dq1, sets, res = qdp_read('newswift/copl_2-10.qdp') ; c_set = ['g','r','b'] ; n_set = ['XMM','Suzaku','Swift']
#dq2, sets, res = qdp_read('newswift/copl+zgB_2-10.qdp') ; c_set = ['g','r','b'] ; n_set = ['XMM','Suzaku','Swift']
### Extrapolated
#dq, sets, res = qdp_read('newswift/copl+zgB_2-10extrapolated.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
### Broadband fits
#dq1, sets, res = qdp_read('newswift/copl+zgB_broadband.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
#dq2, sets, res = qdp_read('newswift/copl+zgB+bb_broadband.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
### Reflection models
#dq1, sets, res = qdp_read('newswift/distant_refl_Rlinked_AXfrozen_PLfree.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
#dq2, sets, res = qdp_read('newswift/blurred_refl_RAXlinked_kdblurfrozen_PLfree.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
### Hi-low state
##dq, sets, res = qdp_read('newswift/suzakuhilo2510.qdp') ; c_set = ['r','r','r'] ; n_set = ['Suzaku','Suzaku','Suzaku']
##dq, sets, res = qdp_read('newswift/hilo_diff.qdp') ; c_set = ['k','r','r'] ; n_set = ['Suzaku','Suzaku','Suzaku']
#dq, sets, res = qdp_read('newswift/hilo_diff_500.qdp') ; c_set = ['r','r','r'] ; n_set = ['Suzaku','Suzaku','Suzaku']
### Individual fits
#dq1, sets, res = qdp_read('newswift/distant_xmm+suz+swift_individualfits.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']
#dq2, sets, res = qdp_read('newswift/blurred_xmm+suz+swift_individualfits.qdp') ; c_set = ['g','r','m','b','c'] ; n_set = ['XMM','Suzaku','PIN','Swift','BAT']


#### Plot up the data
#fig = plt.figure()
#ax = fig.add_subplot(111)
##plt.grid(which='both',linestyle='-',color='0.7')
#for i in range(0,sets):
#    x = i+1
#    if (i == 0):
#        shape = 'D'
#    if (i == 1):
#        shape = 'o'
#    if (i == 2) and (sets < 4):
#        shape = 's'
#    if (i == 2) and (sets > 4):
#        shape = 'o'
#    if (i == 3):
#        shape = 's'
#    if (i == 4):
#        shape = 's'
#    plottertron(dq["e{0}".format(x)], dq["e_err{0}".format(x)], dq["f{0}".format(x)], dq["f_err{0}".format(x)], c_set[i], n_set[i], 'log', 'log', 1, 4, 'marker', shape)
#    if (res == 1):
#        plottertron(dq["e{0}".format(x)], np.zeros(len(dq["m{0}".format(x)])), dq["m{0}".format(x)], np.zeros(len(dq["m{0}".format(x)])), 'k', 'model', 'log', 'log', 0, 3, 'line', shape)
#
#ax.set_xlim(0.3,150.0)
#ax.tick_params(axis='both', which='both', direction='in', top='on', right='on')
#
##plt.savefig('../../LaTeX/IIIZw2/po0_line.png',bbox_inches='tight',dpi=300)



### Plot up the residuals
#if (res == 1):
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
##    ax.set_xlim(0.3,150.0)
#    for i in range(0,sets):
#        x = i+1
#        if (i == 0):
#            shape = 'D'
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
#        plottertron(dq["res_e{0}".format(x)], dq["res_e_err{0}".format(x)], dq["res{0}".format(x)], dq["res_err{0}".format(x)], c_set[i], n_set[i], 'log', 'linear', 0, 2, 'marker', shape)
#        ax.set_xticks([1,2,5,10]) # choose which x locations to have ticks
#        ax.set_xticklabels([1,2,5,10]) # set the labels to display at those ticks
#        ax.get_xaxis().get_minor_formatter().labelOnlyBase = True
#
#    ax.set_xlim(min(dq["res_e1"])-0.03285,max(dq["res_e1"])+0.648)
#    ax.tick_params(axis='both', which='both', direction='in', top='on', right='on')
#    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1)
#
##    plt.savefig('../../LaTeX/IIIZw2/suzaku_hilo.png',bbox_inches='tight',dpi=300)
##plt.savefig('/Users/agonzalez/Dropbox/Graduate/PhD/IIIZw2Paper2017/suzaku_hilo.png', bbox_inches='tight', dpi=300)


###### Gridspec plotting
#if (res == 1):
##    fig = plt.figure()
##    ax = fig.add_subplot(311)
##    ax.axhline(y=1.0, xmin=5e-6, xmax=5e+6, color='k', linewidth=1)
##    plt.figure(figsize=[7,8])
#    plt.figure()
#    gs1 = gridspec.GridSpec(2,1)
#    gs1.update(wspace=0, hspace=0)
#    for i in range(0,sets):
#        x = i+1
#
#        if (i == 0):
#            shape = 'D'
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
#        plottertron(dq1["res_e{0}".format(x)], dq1["res_e_err{0}".format(x)], dq1["res{0}".format(x)], dq1["res_err{0}".format(x)], c_set[i], n_set[i], 'log', 'linear', 0, 3, 'marker', shape)
#
#        ax = plt.subplot(gs1[1])
#        plottertron(dq2["res_e{0}".format(x)], dq2["res_e_err{0}".format(x)], dq2["res{0}".format(x)], dq2["res_err{0}".format(x)], c_set[i], n_set[i], 'log', 'linear', 0, 3, 'marker', shape)
#
##        ax = plt.subplot(gs1[2])
##        plottertron(dq3["res_e{0}".format(x)], dq3["res_e_err{0}".format(x)], dq3["res{0}".format(x)], dq3["res_err{0}".format(x)], c_set[i], n_set[i], 'log', 'linear', 0, 3, 'marker')
#
#    for i in range(0,2):
#        ax = plt.subplot(gs1[i])
#        if (i == 0):
##            ax.text(.3,.8,'cutoffpl',horizontalalignment='center',transform=ax.transAxes)
##            ax.text(.3,.1,'cutoffpl + zgB',horizontalalignment='center',transform=ax.transAxes)
#            ax.text(.3,.1,'cold distant reflection',horizontalalignment='center',transform=ax.transAxes)
#        elif (i == 1):
##            ax.text(.3,.8,'cutoffpl + zgB',horizontalalignment='center',transform=ax.transAxes)
##            ax.text(.3,.1,'cutoffpl + bb + zgB',horizontalalignment='center',transform=ax.transAxes)
#            ax.text(.3,.1,'blurred ionised reflection',horizontalalignment='center',transform=ax.transAxes)
##        elif (i == 2):
##            ax.text(.2,.8,'cutoffpl + zgB',horizontalalignment='center',transform=ax.transAxes)
##        ax.set_xlim(2.0,10.0)
##        ax.set_ylim(0.70,1.6)
#        ax.set_xlim(0.3,150.0)
##        ax.set_ylim(0.40,1.75)
#        ax.set_ylim(0.40,1.60)
#        ax.tick_params(axis='both', which='both', direction='in', top='on', right='on')
#        if (i != 1):
#            ax.set_xticklabels([])
#        ax.axhline(y=1.0, xmin=5e-6, xmax=5e+6, color='k', linestyle='--', linewidth=1)
##        ax.axvline(x=6.4, ymin=5e-6, ymax=5e+6, color='k', linestyle='--', linewidth=1)
##        ax.set_aspect('equal')
#
##plt.savefig('../../LaTeX/IIIZw2/physbroad_res_subplot_line.png',bbox_inches='tight',dpi=300)
##plt.savefig('/Users/agonzalez/Dropbox/Graduate/PhD/IIIZw2Paper2017/xmm+suz_individualfits.png', bbox_inches='tight', dpi=300)


#### Cold reflection
#e1 =  10**(-11.083--10.475)   ;   print e1, " + ", np.sqrt(e1**(2.0)*(0.065**2.0) + e1**(2.0)*(0.014**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.076**2.0) + e1**(2.0)*(0.014**2.0))
#e2 =  10**(-11.083--10.223)   ;   print e2, " + ", np.sqrt(e1**(2.0)*(0.065**2.0) + e1**(2.0)*(0.013**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.076**2.0) + e1**(2.0)*(0.013**2.0))
#e3 =  10**(-11.083--10.355)   ;   print e3, " + ", np.sqrt(e1**(2.0)*(0.065**2.0) + e1**(2.0)*(0.038**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.076**2.0) + e1**(2.0)*(0.038**2.0))
#
#
#### Blurred reflection
#e1 =  10**(-11.108--10.440)   ;   print e1, " + ", np.sqrt(e1**(2.0)*(0.069**2.0) + e1**(2.0)*(0.014**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.043**2.0) + e1**(2.0)*(0.013**2.0))
#e2 =  10**(-11.108--10.222)   ;   print e2, " + ", np.sqrt(e2**(2.0)*(0.069**2.0) + e2**(2.0)*(0.011**2.0)), " / - ", np.sqrt(e2**(2.0)*(0.043**2.0) + e2**(2.0)*(0.011**2.0))
#e3 =  10**(-11.108--10.335)   ;   print e3, " + ", np.sqrt(e3**(2.0)*(0.069**2.0) + e3**(2.0)*(0.019**2.0)), " / - ", np.sqrt(e3**(2.0)*(0.043**2.0) + e3**(2.0)*(0.032**2.0))
##
##
#### Individual blurred
#e1 =  10**(-11.163--10.441)   ;   print e1, " + ", np.sqrt(e1**(2.0)*(0.120**2.0) + e1**(2.0)*(0.016**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.122**2.0) + e1**(2.0)*(0.023**2.0))
#e2 =  10**(-11.108--10.226)   ;   print e2, " + ", np.sqrt(e2**(2.0)*(0.086**2.0) + e2**(2.0)*(0.015**2.0)), " / - ", np.sqrt(e2**(2.0)*(0.110**2.0) + e2**(2.0)*(0.015**2.0))
#e3 =  10**(-11.125--10.331)   ;   print e3, " + ", np.sqrt(e3**(2.0)*(0.309**2.0) + e3**(2.0)*(0.087**2.0)), " / - ", np.sqrt(e3**(2.0)*(0.583**2.0) + e3**(2.0)*(0.139**2.0))

# #### AUG 3
#
# #### Individual cold
# e1 =  10**(-11.156--10.476)   ;   print e1, " + ", np.sqrt(e1**(2.0)*(0.109**2.0) + e1**(2.0)*(0.014**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.144**2.0) + e1**(2.0)*(0.014**2.0))
# e2 =  10**(-11.050--10.226)   ;   print e2, " + ", np.sqrt(e2**(2.0)*(0.086**2.0) + e2**(2.0)*(0.015**2.0)), " / - ", np.sqrt(e2**(2.0)*(0.110**2.0) + e2**(2.0)*(0.015**2.0))
# e3 =  10**(-10.993--10.392)   ;   print e3, " + ", np.sqrt(e3**(2.0)*(0.126**2.0) + e3**(2.0)*(0.054**2.0)), " / - ", np.sqrt(e3**(2.0)*(0.197**2.0) + e3**(2.0)*(0.049**2.0))
# print ""
#
# #### Individual blurred
# e1 =  10**(-11.131--10.441)   ;   print e1, " + ", np.sqrt(e1**(2.0)*(0.083**2.0) + e1**(2.0)*(0.015**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.160**2.0) + e1**(2.0)*(0.027**2.0))
# e2 =  10**(-11.011--10.243)   ;   print e2, " + ", np.sqrt(e2**(2.0)*(0.105**2.0) + e2**(2.0)*(0.031**2.0)), " / - ", np.sqrt(e2**(2.0)*(0.146**2.0) + e2**(2.0)*(0.014**2.0))
# e3 =  10**(-11.088--10.336)   ;   print e3, " + ", np.sqrt(e3**(2.0)*(0.172**2.0) + e3**(2.0)*(0.089**2.0)), " / - ", np.sqrt(e3**(2.0)*(0.540**2.0) + e3**(2.0)*(0.129**2.0))
# print ""

### NEW COLD
e1 =  10**(-10.99--10.48)   ;   print "XMM xilC ", e1, " + ", np.sqrt(e1**(2.0)*(0.10**2.0) + e1**(2.0)*(0.02**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.12**2.0) + e1**(2.0)*(0.02**2.0))
e1 =  10**(-11.11--10.20)   ;   print "Suz xilC ", e1, " + ", np.sqrt(e1**(2.0)*(0.10**2.0) + e1**(2.0)*(0.02**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.13**2.0) + e1**(2.0)*(0.02**2.0))
e1 =  10**(-11.01--10.36)   ;   print "Swi xilC ", e1, " + ", np.sqrt(e1**(2.0)*(0.14**2.0) + e1**(2.0)*(0.05**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.22**2.0) + e1**(2.0)*(0.05**2.0))
print ''
e1 =  10**(-10.91--10.53)   ;   print "XMM kd*xilC ", e1, " + ", np.sqrt(e1**(2.0)*(0.07**2.0) + e1**(2.0)*(0.02**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.09**2.0) + e1**(2.0)*(0.02**2.0))
e1 =  10**(-10.98--10.24)   ;   print "Suz kd*xilC ", e1, " + ", np.sqrt(e1**(2.0)*(0.11**2.0) + e1**(2.0)*(0.03**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.16**2.0) + e1**(2.0)*(0.03**2.0))
e1 =  10**(-11.05--10.35)   ;   print "Suz kd*xilC ", e1, " + ", np.sqrt(e1**(2.0)*(0.17**2.0) + e1**(2.0)*(0.06**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.28**2.0) + e1**(2.0)*(0.06**2.0))
print ''
e1 =  10**(-11.04--10.40)   ;   print "XMM kd*xilB ", e1, " + ", np.sqrt(e1**(2.0)*(0.10**2.0) + e1**(2.0)*(0.02**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.12**2.0) + e1**(2.0)*(0.02**2.0))
e1 =  10**(-10.94--10.21)   ;   print "XMM kd*xilB ", e1, " + ", np.sqrt(e1**(2.0)*(0.10**2.0) + e1**(2.0)*(0.02**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.16**2.0) + e1**(2.0)*(0.02**2.0))
e1 =  10**(-10.96--10.40)   ;   print "XMM kd*xilB ", e1, " + ", np.sqrt(e1**(2.0)*(0.14**2.0) + e1**(2.0)*(0.12**2.0)), " / - ", np.sqrt(e1**(2.0)*(0.12**2.0) + e1**(2.0)*(0.06**2.0))
