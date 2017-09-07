#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: adamg
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import scipy.stats as st
import operator
import time
import os

from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally
plt.rc('font',family='serif')
plt.rc('text',usetex=True)


os.chdir("/Users/agonzalez/Documents/Research/Data/IZw1")
# data = np.genfromtxt("single_sim.txt")
data = np.genfromtxt("multi_sim.txt")
# data = np.genfromtxt("big_sim.txt")
# data = np.genfromtxt("big_sim_aug16.txt")

# os.chdir("/Users/agonzalez/Documents/Research/Data/Mrk1501")
# data = np.genfromtxt("suz.txt")
# data = np.genfromtxt("xmm.txt")

x, y = data[:,0], data[:,1]

xmin, xmax = 2.0, 30.0
ymin, ymax = 0.25, 0.75


## 2-D Kernel Density Estimation -------------------------------------
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

cdict1 = {'blue':   ((0.00, 1.0, 1.0),
                    (0.10, 1.0, 1.0),
                    (0.20, 1.0, 1.0),
                    (0.40, 1.0, 1.0),
                    (0.60, 1.0, 1.0),
                    (0.80, 1.0, 1.0),
                    (1.00, 0.1, 0.1)),

         'green':  ((0.00, 1.0, 1.0),
                    (0.10, 1.0, 1.0),
                    (0.20, 0.8, 0.8),
                    (0.40, 0.6, 0.6),
                    (0.60, 0.4, 0.4),
                    (0.80, 0.2, 0.2),
                    (1.00, 0.0, 0.0)),

         'red':   ((0.00, 1.0, 1.0),
                    (0.10, 1.0, 1.0),
                    (0.20, 0.0, 0.0),
                    (0.40, 0.0, 0.0),
                    (0.60, 0.0, 0.0),
                    (0.80, 0.0, 0.0),
                    (1.00, 0.0, 0.0)),
        }
cmcust = LinearSegmentedColormap('customcmap', cdict1)
## -------------------------------------------------------------------


## 1-D Kernel Density Estimations ------------------------------------
x, y = np.sort(x, kind='mergesort'), np.sort(y, kind='mergesort')
xpdf = st.gaussian_kde(x) ; kx = xpdf(x)
ypdf = st.gaussian_kde(y) ; ky = ypdf(y)
## -------------------------------------------------------------------


plt.figure(figsize=[12.8,9.6])
gs = gridspec.GridSpec(3,3) ; gs.update(wspace=0, hspace=0)

ax1 = plt.subplot(gs[1:, 0])
ax2 = plt.subplot(gs[0, 1:])
ax3 = plt.subplot(gs[1:, 1:])

ax1.plot(ky, y, 'k', linewidth=1.0)
ax1.set_ylim(ymin,ymax)
ax1.set_ylabel(r'Source Velocity /$c$')
ax1.set_xlabel('Density')
ax1.invert_xaxis()
ax1.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('top')
ax1.tick_params(axis='both', which='both', direction='in', bottom='on', right='on')

ax2.plot(x, kx, 'k', linewidth=1.0)
ax2.set_xlim(xmin,xmax)
ax2.set_xlabel(r'Source Height /$r_g$')
ax2.set_ylabel('Density', rotation='270')
ax2.yaxis.set_label_position('right')
ax2.yaxis.set_ticks_position('right')
ax2.xaxis.set_label_position('top')
ax2.xaxis.set_ticks_position('top')
ax2.tick_params(axis='both', which='both', direction='in', bottom='on', left='on')

ax3.set_xlim(xmin, xmax)
ax3.set_ylim(ymin, ymax)
ax3.tick_params(axis='both', which='both', direction='in', top='on', right='on')
ax3.set_xticklabels([]) ; ax3.set_yticklabels([])
# Contourf plot
cfset = ax3.contourf(xx, yy, f, cmap=cmcust) #cmap=plt.cm.get_cmap(scheme))
# cbar4 = plt.colorbar(cfset, pad=0.05)#, ticks=[-0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
# cbar4.ax.set_ylabel('Density', rotation='270', labelpad=25.0)
# Contour plot
cset = ax3.contour(xx, yy, f, colors='k', linewidths=0.5)
# Label plot
# ax3.set_xlabel(r'Source Height /$r_g$')
# ax3.set_ylabel(r'Source Velocity /$c$')

kxmax_index, kxmax_value = max(enumerate(kx), key=operator.itemgetter(1))
kymax_index, kymax_value = max(enumerate(ky), key=operator.itemgetter(1))

ax1.axhline(y=y[kymax_index], color='k', dashes=[5,3], linewidth=1.0)
ax2.axvline(x=x[kxmax_index], color='k', dashes=[5,3], linewidth=1.0)
ax3.axhline(y=y[kymax_index], color='k', dashes=[5,3], linewidth=1.0)
ax3.axvline(x=x[kxmax_index], color='k', dashes=[5,3], linewidth=1.0)

print "Plotting is done!"


# plt.savefig('/Users/agonzalez/Desktop/IZw1_kde_separate.png', bbox_inches='tight', dpi=300)
# plt.savefig('/Users/agonzalez/Desktop/contour_place_holder_colorbar.ps', format='ps', bbox_inches='tight', dpi=300)

plt.show()
