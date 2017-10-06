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
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally
plt.rc('font',family='serif')
plt.rc('text',usetex=True)


# os.chdir("/Users/agonzalez/Documents/Research/Data/IZw1")
# ## data = np.genfromtxt("single_sim.txt")
# # data = np.genfromtxt("multi_sim.txt")
# ## data = np.genfromtxt("big_sim.txt")
# # data = np.genfromtxt("big_sim_aug16.txt")
# data = np.genfromtxt("big_sim_aug18.txt")
#
os.chdir("/Users/agonzalez/Documents/Research/Data/Mrk1501")
## data = np.genfromtxt("suz.txt")
# data = np.genfromtxt("xmm.txt")
data = np.genfromtxt("xmm_big.txt")

x, y = data[:,0], data[:,1]

xmin, xmax = 2.0, 32.0
ymin, ymax = 0.5, 1.0

## binning up each dimesion
x_res, y_res = 25, 25
n_x, n_y = np.zeros(x_res), np.zeros(y_res)
x_bins, y_bins = np.linspace(xmin, xmax, num=x_res), np.linspace(ymin, ymax, num=y_res)
dx, dy = x_bins[1]-x_bins[0], y_bins[1]-y_bins[0]

for i in range (0,len(x)):
    for j in range (0,x_res-1):
        if (x_bins[j] <= x[i] < x_bins[j+1]):
            n_x[j] += 1.

for i in range (0,len(y)):
    for j in range (0,y_res-1):
        if (y_bins[j] <= y[i] < y_bins[j+1]):
            n_y[j] += 1.0

n_x = n_x / max(n_x)
n_y = n_y / max(n_y)

h_xy = np.zeros([x_res,y_res])
for i in range (0,x_res):
    h_xy[i,:] += n_x
for i in range (0, y_res):
    h_xy[:,i] += n_y
X, Y = np.meshgrid(x_bins,y_bins)
print "Done meshgrid"
# x_bins += dx
# y_bins += dy

# plt.figure()
# plt.step(x_bins, n_x, where='mid')
#
# plt.figure()
# plt.step(y_bins, n_y, where='mid')
#
# plt.show()

cdict1 = {'blue':  ((0.00, 1.0, 1.0),
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

         'red':    ((0.00, 1.0, 1.0),
                    (0.10, 1.0, 1.0),
                    (0.20, 0.0, 0.0),
                    (0.40, 0.0, 0.0),
                    (0.60, 0.0, 0.0),
                    (0.80, 0.0, 0.0),
                    (1.00, 0.0, 0.0)),
        }
cmcust = LinearSegmentedColormap('customcmap', cdict1)


## Plot up all the results -------------------------------------------
plt.figure(figsize=[12.8,9.6])
gs = gridspec.GridSpec(3,3) ; gs.update(wspace=0, hspace=0)

ax1 = plt.subplot(gs[1:, 2])
ax2 = plt.subplot(gs[0, :-1])
ax3 = plt.subplot(gs[1:, :-1])

# ax1.step(n_y, y_bins, where='mid')
ax1.plot(n_y, y_bins, '-k')
ax1.scatter(n_y, y_bins, s=10.0, c='k')
# ax1.fill_betweenx(y=y_bins, x1=n_y, x2=0, color='b', alpha=0.25)
ax1.set_ylim(ymin,ymax)
# ax1.set_ylabel(r'Source Velocity /$c$')
# ax1.set_xlabel('Number')
# ax1.invert_xaxis()
# ax1.xaxis.set_label_position('top')
# ax1.xaxis.set_ticks_position('top')
ax1.set_xlim(left=0.0)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.tick_params(axis='both', which='both', direction='in', bottom='on', right='on')

# ax2.step(x_bins, n_x, where='mid')
ax2.plot(x_bins, n_x, '-k')
ax2.scatter(x_bins, n_x, s=10.0, c='k')
# ax2.fill_between(x=x_bins, y1=n_x, y2=0, color='b', alpha=0.25)
ax2.set_xlim(xmin,xmax-xmin)
# ax2.set_xlabel(r'Source Height /$r_g$')
# ax2.set_ylabel('Number', rotation='270', labelpad=20.0)
# ax2.yaxis.set_label_position('right')
# ax2.yaxis.set_ticks_position('right')
# ax2.xaxis.set_label_position('top')
# ax2.xaxis.set_ticks_position('top')
ax2.set_ylim(bottom=0.0)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.tick_params(axis='both', which='both', direction='in', bottom='on', left='on')

# # ax3.set_xticklabels([]) ; ax3.set_yticklabels([])
# # Hist2d plot
# cfset = ax3.hist2d(x,y,bins=[x_res,y_res], range=[[xmin,xmax],[ymin,ymax]], cmap=cmcust)
# for i in range (0,x_res):
#     ax3.axvline(x=x_bins[i], color='k', linewidth=1.0)
# for i in range (0,y_res):
#     ax3.axhline(y=y_bins[i], color='k', linewidth=1.0)

ax3.contourf(X,Y,h_xy, cmap=cmcust)
kxmax_index, kxmax_value = max(enumerate(n_x), key=operator.itemgetter(1))
kymax_index, kymax_value = max(enumerate(n_y), key=operator.itemgetter(1))
ax1.axhline(y=y_bins[kymax_index], color='k', dashes=[5,3], linewidth=1.0)
ax2.axvline(x=x_bins[kxmax_index], color='k', dashes=[5,3], linewidth=1.0)
ax3.axhline(y=y_bins[kymax_index], color='k', dashes=[5,3], linewidth=1.0)
ax3.axvline(x=x_bins[kxmax_index], color='k', dashes=[5,3], linewidth=1.0)
ax3.scatter(x_bins[kxmax_index], y_bins[kymax_index], c='r', s=15.0)
print "height   = ", x_bins[kxmax_index]
print "velocity = ", y_bins[kymax_index]

# Label plot
# ax3.invert_xaxis()
ax3.set_xlim(xmin, xmax-xmin)
ax3.set_ylim(ymin, ymax)
ax3.set_xlabel(r'Source Height /$r_g$')
ax3.set_ylabel(r'Source Velocity /$c$')
# ax3.set_ylabel(r'Source Velocity /$c$', rotation='270', labelpad=20.0)
# ax3.yaxis.set_label_position('right')
# ax3.yaxis.set_ticks_position('right')
ax3.tick_params(axis='both', which='both', direction='in', top='on', left='on')


## ------------------------------------------------------------------

# ####################################################################################################
# Compute the escape velocity for a black hole of mass M at a height R above the black hole
def vesc_calc(G,M,R,c):
    v = np.sqrt((2.0*G*M)/R)/c

    return v

G = 6.674e-11
c = 2.998e8
M_sun = 1.989e30

# plt.figure()
# ax = plt.subplot(111)
col = ['r','r','r']

res = 50
Vesc = np.zeros([5,res])
R = np.zeros([5,res])

for j in range (0,3):
    ## I Zw 1
    # if (j==0):
    #     M_bh = pow(10.0, 7.30)*M_sun ; name = 'Negrete et al. (2012)'
    #     r_g0 = (G*M_bh)/(c**2.0)
    # if (j==1):
    #     M_bh = pow(10.0, 7.30+0.23)*M_sun ; name = 'Mass + error'
    # if (j==2):
    #     M_bh = pow(10.0, 7.30-0.19)*M_sun ; name = 'Mass -- error'

    ## III Zw 2
    if (j==0):
        M_bh = pow(10.0, 8.03)*M_sun ; name = 'van den Bosch (2016)'
        r_g0 = (G*M_bh)/(c**2.0)
    if (j==1):
        M_bh = pow(10.0, 8.03+0.26)*M_sun ; name = '+'
    if (j==2):
        M_bh = pow(10.0, 8.03-0.26)*M_sun ; name = '--'

    R_s = (2.0*G*M_bh)/(c**2.0)
    r_g = (G*M_bh)/(c**2.0)

    R[j][:] = np.logspace(start=np.log10(1.01*R_s), stop=np.log10(1000.0*r_g), num=res)

    for i in range (0,res):
        Vesc[j][i] = vesc_calc(G,M_bh,R[j][i],c)

    # print "Mass of I Zw 1 BH [kg]   = ", M_bh
    # print "Schwarzschild radius [m] = ", R_s
    # print "Gravitationl radius [m]  = ", r_g
    R[j][:] = R[j][:]/r_g0

    if (j!=0):
        ax3.plot(R[j][:],Vesc[j][:], color=col[j], dashes=[5,3], alpha=0.75, label=name)
    elif (j==0):
        ax3.plot(R[j][:],Vesc[j][:], color=col[j], alpha=0.75, label=name)

for i in range (0,res):
    R[3][i] = abs(R[0][i]-R[1][i])
    R[4][i] = abs(R[0][i]-R[2][i])
# ####################################################################################################

# plt.savefig('/Users/agonzalez/Desktop/IZw1_kde_separate_histogram.png', bbox_inches='tight', dpi=300)
# plt.savefig('/Users/agonzalez/Desktop/IIIZw2_kde_separate_xmm.png', bbox_inches='tight', dpi=300)
# plt.savefig('/Users/agonzalez/Desktop/contour_place_holder_colorbar.ps', format='ps', bbox_inches='tight', dpi=300)

plt.show()
