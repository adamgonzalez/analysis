#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: agonzalez
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

from matplotlib.ticker import ScalarFormatter
from matplotlib import gridspec

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally
plt.rc('font',family='serif')
plt.rc('text',usetex=True)


os.chdir("/Users/agonzalez/Documents/Research/data/IZw1")


### Read in the data1 from the .qdp files made by wdata1 in iplot on XSPEC
# data1 = np.genfromtxt('sphere_temp.qdp', skip_header=3)
# data2 = np.genfromtxt('jet_temp.qdp', skip_header=3)

data1 = np.genfromtxt('sphere_pexmon.qdp', skip_header=3)
data2 = np.genfromtxt('jet_pexmon.qdp', skip_header=3)
# data3 = np.genfromtxt('sphere+jet_pexmon.qdp', skip_header=3)
data3 = np.genfromtxt('lcg_sphere+jet.qdp', skip_header=3)


fig = plt.figure(1)
ax0 = fig.add_subplot(111)
ax0.spines['top'].set_color('none')
ax0.spines['bottom'].set_color('none')
ax0.spines['left'].set_color('none')
ax0.spines['right'].set_color('none')
ax0.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

# gs1 = gridspec.GridSpec(2,1)
# gs1.update(wspace=0, hspace=0)
# ax = plt.subplot(gs1[0]) ; ax1 = plt.subplot(gs1[1])
ax = fig.add_subplot(211) ; ax1 = fig.add_subplot(212)
plt.subplots_adjust(wspace=0, hspace=0)

ax.set_xscale('log')
ax.set_yscale('log')
# ax.axvline(x=6.4, color='k', dashes=[5,3], linewidth=1.0)
ax.errorbar(data1[:,0], data1[:,2], xerr=data1[:,1], yerr=data1[:,3], fmt='+', markersize=0.5, color='0.5', ecolor='0.5', elinewidth=1.0, capsize=None, capthick=None, alpha=1.0, label='Unfolded Spectrum')
ax.plot(data1[:,0], data1[:,4], color='r', dashes=[5,3], linewidth=3.0, alpha=1.0, label='Sphere')
ax.plot(data2[:,0], data2[:,4], color='b', dashes=[1.5,1.5], linewidth=3.0, alpha=1.0, label='Jet')
ax.set_xlim(2.0,10.0)
ax.set_ylim(0.0018,0.003)
ax.set_xticks([2,5,10]) # choose which x locations to have ticks
ax.set_xticklabels([])#[2,5,10]) # set the labels to display at those ticks
ax.get_xaxis().get_minor_formatter().labelOnlyBase = True
ax.set_yticks([2e-3,2.5e-3]) # choose which x locations to have ticks
ax.set_yticklabels([2e-3,2.5e-3], verticalalignment='center', rotation='vertical') # set the labels to display at those ticks
ax.get_yaxis().get_minor_formatter().labelOnlyBase = True
ax.tick_params(axis='both', which='both', direction='in', top='on', right='on')
ax.legend(loc=1, ncol=3, labelspacing=0.1, fontsize=14, handletextpad=0.25, fancybox=False, frameon=False)
# ax.get_legend()


ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.axvline(x=6.4, color='k', dashes=[5,3], linewidth=1.0)
ax1.errorbar(data3[:,0], data3[:,2], xerr=data3[:,1], yerr=data3[:,3], fmt='+', markersize=0.5, color='0.5', ecolor='0.5', elinewidth=1.0, capsize=None, capthick=None, alpha=1.0, label='Unfolded Spectrum')
ax1.plot(data3[:,0], data3[:,4], color='g', linestyle='-', linewidth=3.0, alpha=1.0, label='Sphere+Jet')
ax1.set_xlim(2.0,10.0)
ax1.set_ylim(0.0018,0.003)
ax1.set_xticks([2,5,10]) # choose which x locations to have ticks
ax1.set_xticklabels([2,5,10]) # set the labels to display at those ticks
ax1.get_xaxis().get_minor_formatter().labelOnlyBase = True
ax1.set_yticks([2e-3,2.5e-3]) # choose which x locations to have ticks
ax1.set_yticklabels([2e-3,2.5e-3], verticalalignment='center', rotation='vertical') # set the labels to display at those ticks
ax1.get_yaxis().get_minor_formatter().labelOnlyBase = True
# ax1.set_xlabel(r'Energy (keV)')
ax1.tick_params(axis='both', which='both', direction='in', top='on', right='on')
ax1.legend(loc=1, ncol=3, labelspacing=0.1, fontsize=14, handletextpad=0.25, fancybox=False, frameon=False)
# ax1.get_legend()


ax0.set_xlabel('Energy (keV)')
ax0.set_ylabel('keV$^{2}$ (Photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$)')


plt.savefig('unfolded_spec.ps', format='ps', bbox_inches='tight', dpi=300)

plt.show()
