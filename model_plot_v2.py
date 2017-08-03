#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:56:12 2017

@author: agonzalez
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math
from matplotlib.ticker import ScalarFormatter
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally


####################################################################################################
# read in the qdp data files
def qdp_read(filename):
    data = np.genfromtxt(filename)

    new_set = []
    for i in range (0, len(data)):
        if (math.isnan(data[i,0]) == 1):
            new_set.append(i)

    r = 0
    if ((len(new_set)+1)/2. != int((len(new_set)+1)/2)):
        n_sets = len(new_set)+1
    else:
        n_sets = (len(new_set)+1)/2
        r = 1


    if (n_sets == 1) and (r==1):
        d_e1     = data[:new_set[0]-1,0]    ;   d_r_e1     = data[new_set[0]+1:,0]
        d_e_err1 = data[:new_set[0]-1,1]    ;   d_r_e_err1 = data[new_set[0]+1:,1]
        d_f1     = data[:new_set[0]-1,2]    ;   d_r_f1     = data[new_set[0]+1:,2]
        d_f_err1 = data[:new_set[0]-1,3]    ;   d_r_f_err1 = data[new_set[0]+1:,3]
        m1 = data[:new_set[0]-1,4]

        return (d_e1, d_e_err1, d_f1, d_f_err1, d_r_e1, d_r_e_err1, d_r_f1, d_r_f_err1, m1, n_sets)

    elif (n_sets == 2) and (r==1):
        d_e1     = data[:new_set[0]-1,0]    ;   d_r_e1     = data[new_set[1]+1:new_set[2]-1,0]
        d_e_err1 = data[:new_set[0]-1,1]    ;   d_r_e_err1 = data[new_set[1]+1:new_set[2]-1,1]
        d_f1     = data[:new_set[0]-1,2]    ;   d_r_f1     = data[new_set[1]+1:new_set[2]-1,2]
        d_f_err1 = data[:new_set[0]-1,3]    ;   d_r_f_err1 = data[new_set[1]+1:new_set[2]-1,3]
        m1 = data[:new_set[0]-1,4]

        d_e2     = data[new_set[0]+1:new_set[1]-1,0]    ;   d_r_e2     = data[new_set[2]+1:,0]
        d_e_err2 = data[new_set[0]+1:new_set[1]-1,1]    ;   d_r_e_err2 = data[new_set[2]+1:,1]
        d_f2     = data[new_set[0]+1:new_set[1]-1,2]    ;   d_r_f2     = data[new_set[2]+1:,2]
        d_f_err2 = data[new_set[0]+1:new_set[1]-1,3]    ;   d_r_f_err2 = data[new_set[2]+1:,3]
        m2 = data[new_set[0]+1:new_set[1]-1,4]

        return (d_e1, d_e_err1, d_f1, d_f_err1, d_r_e1, d_r_e_err1, d_r_f1, d_r_f_err1, m1,
                d_e2, d_e_err2, d_f2, d_f_err2, d_r_e2, d_r_e_err2, d_r_f2, d_r_f_err2, m2,
                n_sets)

    elif (n_sets == 3) and (r==1):
        d_e1     = data[:new_set[0]-1,0]    ;   d_r_e1     = data[new_set[2]+1:new_set[3]-1,0]
        d_e_err1 = data[:new_set[0]-1,1]    ;   d_r_e_err1 = data[new_set[2]+1:new_set[3]-1,1]
        d_f1     = data[:new_set[0]-1,2]    ;   d_r_f1     = data[new_set[2]+1:new_set[3]-1,2]
        d_f_err1 = data[:new_set[0]-1,3]    ;   d_r_f_err1 = data[new_set[2]+1:new_set[3]-1,3]
        m1 = data[:new_set[0]-1,4]

        d_e2     = data[new_set[0]+1:new_set[1]-1,0]    ;   d_r_e2     = data[new_set[3]+1:new_set[4]-1,0]
        d_e_err2 = data[new_set[0]+1:new_set[1]-1,1]    ;   d_r_e_err2 = data[new_set[3]+1:new_set[4]-1,1]
        d_f2     = data[new_set[0]+1:new_set[1]-1,2]    ;   d_r_f2     = data[new_set[3]+1:new_set[4]-1,2]
        d_f_err2 = data[new_set[0]+1:new_set[1]-1,3]    ;   d_r_f_err2 = data[new_set[3]+1:new_set[4]-1,3]
        m2 = data[new_set[0]+1:new_set[1]-1,4]

        d_e3     = data[new_set[1]+1:new_set[2]-1,0]    ;   d_r_e3     = data[new_set[4]+1:,0]
        d_e_err3 = data[new_set[1]+1:new_set[2]-1,1]    ;   d_r_e_err3 = data[new_set[4]+1:,1]
        d_f3     = data[new_set[1]+1:new_set[2]-1,2]    ;   d_r_f3     = data[new_set[4]+1:,2]
        d_f_err3 = data[new_set[1]+1:new_set[2]-1,3]    ;   d_r_f_err3 = data[new_set[4]+1:,3]
        m3 = data[new_set[1]+1:new_set[2]-1,4]

        return (d_e1, d_e_err1, d_f1, d_f_err1, d_r_e1, d_r_e_err1, d_r_f1, d_r_f_err1, m1,
                d_e2, d_e_err2, d_f2, d_f_err2, d_r_e2, d_r_e_err2, d_r_f2, d_r_f_err2, m2,
                d_e3, d_e_err3, d_f3, d_f_err3, d_r_e3, d_r_e_err3, d_r_f3, d_r_f_err3, m3,
                n_sets)

    elif (n_sets == 5) and (r==0):
        d_e1     = data[:new_set[0]-1,0]    ;   d_e_err1 = data[:new_set[0]-1,1]
        d_f1     = data[:new_set[0]-1,2]    ;   d_f_err1 = data[:new_set[0]-1,3]

        d_e2     = data[new_set[0]+1:new_set[1]-1,0]    ;   d_e_err2 = data[new_set[0]+1:new_set[1]-1,1]
        d_f2     = data[new_set[0]+1:new_set[1]-1,2]    ;   d_f_err2 = data[new_set[0]+1:new_set[1]-1,3]

        d_e3     = data[new_set[1]+1:new_set[2]-1,0]    ;   d_e_err3 = data[new_set[1]+1:new_set[2]-1,1]
        d_f3     = data[new_set[1]+1:new_set[2]-1,2]    ;   d_f_err3 = data[new_set[1]+1:new_set[2]-1,3]

        d_e4     = data[new_set[2]+1:new_set[3]-1,0]    ;   d_e_err4 = data[new_set[2]+1:new_set[3]-1,1]
        d_f4     = data[new_set[2]+1:new_set[3]-1,2]    ;   d_f_err4 = data[new_set[2]+1:new_set[3]-1,3]

        d_e5     = data[new_set[3]+1:,0]    ;   d_e_err5 = data[new_set[3]+1:,1]
        d_f5     = data[new_set[3]+1:,2]    ;   d_f_err5 = data[new_set[3]+1:,3]

        return (d_e1, d_e_err1, d_f1, d_f_err1,
                d_e2, d_e_err2, d_f2, d_f_err2,
                d_e3, d_e_err3, d_f3, d_f_err3,
                d_e4, d_e_err4, d_f4, d_f_err4,
                d_e5, d_e_err5, d_f5, d_f_err5,
                n_sets)
####################################################################################################


####################################################################################################
# Plot x and y with labels (trying to do it)
def plottertron(x, x_error, y, y_error, line_color, line_label, xscale, yscale, legyn, legloc, style):
    ax.set_xlabel('Energy ($\mathrm{keV}$)')
    ax.set_ylabel('Ratio')
    if (xscale == 'log'):
        ax.set_xscale('log')
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if (yscale == 'log'):
#        ax.set_ylabel('Normalized Counts ($\mathrm{s^{-1}}$ $\mathrm{keV^{-1}}$)')
        ax.set_ylabel('$\mathrm{keV^{2}}$ (Photons $\mathrm{cm^{-2}}$ $\mathrm{s^{-1}}$ $\mathrm{keV^{-1}}$)')
        ax.set_yscale('log')
    if (style == 'line'):
        ax.errorbar(x, y, xerr=x_error, yerr=y_error, ecolor=line_color, capthick=1, color=line_color, label=line_label)
    elif (style == 'marker'):
        ax.errorbar(x, y, xerr=x_error, yerr=y_error, fmt='o', markersize=1, ecolor=line_color, capthick=1, linewidth = 1, color=line_color, label=line_label)
    if (legyn == 1):
        ax.legend(loc=legloc, labelspacing=0.1, fontsize=16)
    plt.show()
####################################################################################################


### Read in the data from the .qdp files made by wdata in iplot on XSPEC

## XMM ####################################################################################################
#(e1, e_err1, f1, f_err1, r_e1, r_e_err1, r_f1, r_f_err1, m1,
#e2, e_err2, f2, f_err2, r_e2, r_e_err2, r_f2, r_f_err2, m2,
#sets) = qdp_read('xmm_BAT_tbabspo_2-10extr.qdp') ; d1 = 'XMM' ; d2 = 'BAT'

#(e1, e_err1, f1, f_err1, r_e1, r_e_err1, r_f1, r_f_err1, m1,
#e2, e_err2, f2, f_err2, r_e2, r_e_err2, r_f2, r_f_err2, m2,
#sets) = qdp_read('xmm_BAT_tbabspo_03-150.qdp') ; d1 = 'XMM' ; d2 = 'BAT'

#(e1, e_err1, f1, f_err1, r_e1, r_e_err1, r_f1, r_f_err1, m1,
#e2, e_err2, f2, f_err2, r_e2, r_e_err2, r_f2, r_f_err2, m2,
#sets) = qdp_read('xmm_BAT_tbabspo_reflionx_03-150.qdp') ; d1 = 'XMM' ; d2 = 'BAT'

#(e1, e_err1, f1, f_err1, r_e1, r_e_err1, r_f1, r_f_err1, m1,
#e2, e_err2, f2, f_err2, r_e2, r_e_err2, r_f2, r_f_err2, m2,
#sets) = qdp_read('xmm_BAT_tbabspo_kdblurreflionx_03-150.qdp') ; d1 = 'XMM' ; d2 = 'BAT'


## Suzaku ####################################################################################################
#(e1, e_err1, f1, f_err1, r_e1, r_e_err1, r_f1, r_f_err1, m1,
#e2, e_err2, f2, f_err2, r_e2, r_e_err2, r_f2, r_f_err2, m2,
#e3, e_err3, f3, f_err3, r_e3, r_e_err3, r_f3, r_f_err3, m3,
#sets) = qdp_read('suzaku_PIN_BAT_tbabspo_2-10extrapolated.qdp') ; d1 = 'XIS0+3' ; d2 = 'PIN' ; d3 = 'BAT'

#(e1, e_err1, f1, f_err1, r_e1, r_e_err1, r_f1, r_f_err1, m1,
#e2, e_err2, f2, f_err2, r_e2, r_e_err2, r_f2, r_f_err2, m2,
#e3, e_err3, f3, f_err3, r_e3, r_e_err3, r_f3, r_f_err3, m3,
#sets) = qdp_read('suzaku_PIN_BAT_tbabspo_07-150.qdp') ; d1 = 'XIS0+3' ; d2 = 'PIN' ; d3 = 'BAT'

#(e1, e_err1, f1, f_err1, r_e1, r_e_err1, r_f1, r_f_err1, m1,
#e2, e_err2, f2, f_err2, r_e2, r_e_err2, r_f2, r_f_err2, m2,
#e3, e_err3, f3, f_err3, r_e3, r_e_err3, r_f3, r_f_err3, m3,
#sets) = qdp_read('suzaku_PIN_BAT_tbabspo_reflionx_07-150.qdp') ; d1 = 'XIS0+3' ; d2 = 'PIN' ; d3 = 'BAT'


## SWIFT ####################################################################################################
#(e1, e_err1, f1, f_err1, r_e1, r_e_err1, r_f1, r_f_err1, m1,
#e2, e_err2, f2, f_err2, r_e2, r_e_err2, r_f2, r_f_err2, m2,
#sets) = qdp_read('swift_BAT_tbabspo_2-10extr.qdp') ; d1 = 'SWIFT' ; d2 = 'BAT'

#(e1, e_err1, f1, f_err1, r_e1, r_e_err1, r_f1, r_f_err1, m1,
#e2, e_err2, f2, f_err2, r_e2, r_e_err2, r_f2, r_f_err2, m2,
#sets) = qdp_read('swift_BAT_tbabspo_03-150.qdp') ; d1 = 'SWIFT' ; d2 = 'BAT'

#(e1, e_err1, f1, f_err1, r_e1, r_e_err1, r_f1, r_f_err1, m1,
#e2, e_err2, f2, f_err2, r_e2, r_e_err2, r_f2, r_f_err2, m2,
#sets) = qdp_read('swift_BAT_tbabspo_reflionx_03-150.qdp') ; d1 = 'SWIFT' ; d2 = 'BAT'


## XMM Suzaku SWIFT ####################################################################################################
#(e1, e_err1, f1, f_err1, r_e1, r_e_err1, r_f1, r_f_err1, m1,
#e2, e_err2, f2, f_err2, r_e2, r_e_err2, r_f2, r_f_err2, m2,
#e3, e_err3, f3, f_err3, r_e3, r_e_err3, r_f3, r_f_err3, m3,
#sets) = qdp_read('xmm_suzaku_swift_tbabspo.qdp') ; d1 = 'XMM' ; d2 = 'Suzkau' ; d3 = 'SWIFT'


#(e1, e_err1, f1, f_err1, r_e1, r_e_err1, r_f1, r_f_err1, m1,
#sets) = qdp_read('suzakuhilo.qdp') ; d1 = 'Suzaku'


data = np.genfromtxt('distant_constant_reflection.qdp', skip_header = 3)


#### Plot up the data
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plottertron(e1, e_err1, f1, f_err1, 'r', d1, 'log', 'log', 0, 3, 'marker')
#plottertron(e1, np.zeros(len(e1)), m1, np.zeros(len(e1)), 'k', 'model', 'log', 'log', 0, 3, 'line')
#if (sets >= 2):
#    plottertron(e2, e_err2, f2, f_err2, 'b', d2, 'log', 'log', 0, 3, 'marker')
#    plottertron(e2, np.zeros(len(e2)), m2, np.zeros(len(e2)), 'k', 'model', 'log', 'log', 0, 3, 'line')
#if (sets >= 3):
#    plottertron(e3, e_err3, f3, f_err3, 'g', d3, 'log', 'log', 0, 3, 'marker')
#    plottertron(e3, np.zeros(len(e3)), m3, np.zeros(len(e3)), 'k', 'model', 'log', 'log', 0, 3, 'line')
#plt.grid(which='both',linestyle='-',color='0.7')
#
##plt.savefig('swift_BAT_tbabspo_reflionx_03-150.png',bbox_inches='tight')
#
#
#### Plot up the residuals
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.axhline(y=1.0, xmin=0.005, xmax=150.0, color='k', linewidth=2)
#plottertron(r_e1, r_e_err1, r_f1, r_f_err1, 'r', d1, 'log', 'linear', 1, 3, 'marker')
#if (sets >= 2):
#    plottertron(r_e2, r_e_err2, r_f2, r_f_err2, 'b', d2, 'log', 'linear', 1, 3, 'marker')
#if (sets >= 3):
#    plottertron(r_e3, r_e_err3, r_f3, r_f_err3, 'g', d3, 'log', 'linear', 1, 3, 'marker')
#
#
##plt.savefig('swift_BAT_tbabspo_reflionx_03-150_residuals.png',bbox_inches='tight')



### BIG DATA SET ####################################################################################################
#(e1, e_err1, f1, f_err1,
#e2, e_err2, f2, f_err2,
#e3, e_err3, f3, f_err3,
#e4, e_err4, f4, f_err4,
#e5, e_err5, f5, f_err5,
#sets) = qdp_read('po0.qdp') ; d1 = 'XMM' ; d2 = 'Suzaku' ; d3 = 'PIN' ; d4 = 'Swift' ; d5 = 'BAT'
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.grid(which='both',linestyle='-',color='0.7')
#plottertron(e1, e_err1, f1, f_err1, 'k', d1, 'log', 'log', 0, 4, 'marker')
##plt.savefig('xmm_po0.png', bbox_inches='tight')
#
#plottertron(e2, e_err2, f2, f_err2, 'r', d2, 'log', 'log', 0, 4, 'marker')
#plottertron(e3, e_err3, f3, f_err3, 'm', d3, 'log', 'log', 0, 4, 'marker')
##plt.savefig('xmm_suzpin_po0.png', bbox_inches='tight')
#
#plottertron(e4, e_err4, f4, f_err4, 'b', d4, 'log', 'log', 0, 4, 'marker')
#plottertron(e5, e_err5, f5, f_err5, 'c', d5, 'log', 'log', 1, 4, 'marker')
##plt.savefig('xmm_suzpin_swfitbat_po0.png', bbox_inches='tight')
