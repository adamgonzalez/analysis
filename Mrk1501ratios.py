#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:19:58 2017

@author: agonzalez
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
from matplotlib.ticker import ScalarFormatter
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally
plt.rc('font',family='serif')
plt.rc('text',usetex=True)

xmm_c = np.genfromtxt('newswift/res_xmm_copl.qdp', skip_header=3)
xmm_cz = np.genfromtxt('newswift/res_xmm_copl+zgB.qdp', skip_header=3)
suz_c = np.genfromtxt('newswift/res_suz_copl.qdp', skip_header=3)
suz_cz = np.genfromtxt('newswift/res_suz_copl+zgN.qdp', skip_header=3)
swift_c = np.genfromtxt('newswift/res_swift_copl.qdp', skip_header=3)
swift_cz = np.genfromtxt('newswift/res_swift_copl+zgB.qdp', skip_header=3)

fig = plt.figure()
gs1 = gridspec.GridSpec(2,1)
gs1.update(wspace=0, hspace=0)

ax1 = plt.subplot(gs1[0])
ax1.set_xlabel('Energy (keV)') ; ax1.set_ylabel('Ratio') ; ax1.set_xscale('log')
#ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.set_xticks([4,5,6,7]) # choose which x locations to have ticks
ax1.set_xticklabels([]) # set the labels to display at those ticks
ax1.set_yticks([1,1.3])
ax1.set_yticklabels([1,1.3])
ax1.get_xaxis().get_minor_formatter().labelOnlyBase = True

ax1.axhline(y=1.0, color='k', linewidth=1.0, dashes=[5,3])
ax1.axvline(x=6.4/(1+0.089338), color='k', linewidth=1.0, dashes=[5,3])
ax1.errorbar(x=xmm_c[:,0], y=xmm_c[:,2], xerr=xmm_c[:,1], yerr=xmm_c[:,3], fmt='D', color='g', ecolor='g', capthick=1, linewidth=1, markersize=0)
ax1.errorbar(x=suz_c[:,0], y=suz_c[:,2], xerr=suz_c[:,1], yerr=suz_c[:,3], fmt='o', color='r', ecolor='r', capthick=1, linewidth=1, markersize=0)
ax1.text(.20,.7,'cutoffpl',horizontalalignment='center',transform=ax1.transAxes)
#ax1.errorbar(x=swift_c[:,0], y=swift_c[:,2], xerr=swift_c[:,1], yerr=swift_c[:,3], fmt='o', color='b', ecolor='b', capthick=1, linewidth=1, markersize=0)
#ax1.fill_between(x=xmm_c[:,0], y1=xmm_c[:,2]-xmm_c[:,3], y2=xmm_c[:,2]+xmm_c[:,3], color='g', alpha=0.75)
#ax1.fill_between(x=suz_c[:,0], y1=suz_c[:,2]-suz_c[:,3], y2=suz_c[:,2]+suz_c[:,3], color='r', alpha=0.75)
#ax1.fill_between(x=swift_c[:,0], y1=swift_c[:,2]-swift_c[:,3], y2=swift_c[:,2]+swift_c[:,3], color='b', alpha=0.75)

#ax1.set_xlim(0.3,10.0)
ax1.set_xlim(4.0,7.0)
ax1.set_ylim(0.80,1.4)
ax1.tick_params(axis='both', which='both', direction='in', top='on', right='on')


ax1 = plt.subplot(gs1[1])
ax1.set_xlabel('Energy (keV)') ; ax1.set_ylabel('Ratio') ; ax1.set_xscale('log')
#ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.set_xticks([4,5,6,7]) # choose which x locations to have ticks
ax1.set_xticklabels([4,5,6,7]) # set the labels to display at those ticks
ax1.set_yticks([1,1.3])
ax1.set_yticklabels([1,1.3])
ax1.get_xaxis().get_minor_formatter().labelOnlyBase = True

ax1.axhline(y=1.0, color='k', linewidth=1.0, dashes=[5,3])
ax1.axvline(x=6.4/(1+0.089338), color='k', linewidth=1.0, dashes=[5,3])
ax1.errorbar(x=xmm_cz[:,0], y=xmm_cz[:,2], xerr=xmm_cz[:,1], yerr=xmm_cz[:,3], fmt='D', color='g', ecolor='g', capthick=1, linewidth=1, markersize=0)
ax1.errorbar(x=suz_cz[:,0], y=suz_cz[:,2], xerr=suz_cz[:,1], yerr=suz_cz[:,3], fmt='o', color='r', ecolor='r', capthick=1, linewidth=1, markersize=0)
ax1.text(.20,.7,'cutoffpl+zg',horizontalalignment='center',transform=ax1.transAxes)
#ax1.errorbar(x=swift_cz[:,0], y=swift_cz[:,2], xerr=swift_cz[:,1], yerr=swift_cz[:,3], fmt='o', color='b', ecolor='b', capthick=1, linewidth=1, markersize=0)
#ax1.fill_between(x=xmm_cz[:,0], y1=xmm_cz[:,2]-xmm_cz[:,3], y2=xmm_cz[:,2]+xmm_c[:,3], color='g', alpha=0.75)
#ax1.fill_between(x=suz_cz[:,0], y1=suz_cz[:,2]-suz_cz[:,3], y2=suz_cz[:,2]+suz_c[:,3], color='r', alpha=0.75)
#ax1.fill_between(x=swift_cz[:,0], y1=swift_cz[:,2]-swift_cz[:,3], y2=swift_cz[:,2]+swift_c[:,3], color='b', alpha=0.75)

#ax1.set_xlim(0.3,10.0)
ax1.set_xlim(4.0,7.0)
ax1.set_ylim(0.80,1.4)
ax1.tick_params(axis='both', which='both', direction='in', top='on', right='on')

#plt.savefig('/Users/agonzalez/Desktop/ironline.png', bbox_inches='tight', dpi=300)
######plt.savefig('/Users/agonzalez/Dropbox/Graduate/PhD/IIIZw2Paper2017/xmm+suz_individualfits.png', bbox_inches='tight', dpi=300)
