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
## data = np.genfromtxt("single_sim.txt")
data = np.genfromtxt("multi_sim.txt")
## data = np.genfromtxt("big_sim.txt")
# data = np.genfromtxt("big_sim_aug16.txt")
#
## os.chdir("/Users/agonzalez/Documents/Research/Data/Mrk1501")
## data = np.genfromtxt("suz.txt")
## data = np.genfromtxt("xmm.txt")

x, y = data[:,0], data[:,1]

xmin, xmax = 2.0, 30.0
ymin, ymax = 0.25, 0.75


# ## 2-D Kernel Density Estimation -------------------------------------
t0 = time.time()
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)
t1 = time.time()

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
# ## -------------------------------------------------------------------


## 1-D Kernel Density Estimations ------------------------------------
t2 = time.time()
x, y = np.sort(x, kind='mergesort'), np.sort(y, kind='mergesort')
t3 = time.time()
t4 = time.time()
xpdf = st.gaussian_kde(x) ; kx = xpdf(x)
t5 = time.time()
t6 = time.time()
ypdf = st.gaussian_kde(y) ; ky = ypdf(y)
t7 = time.time()

# kx = np.genfromtxt('1d_xkde.txt')
# ky = np.genfromtxt('1d_ykde.txt')
## -------------------------------------------------------------------


## Plot up all the results -------------------------------------------
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
# ax1.xaxis.set_label_position('top')
# ax1.xaxis.set_ticks_position('top')
ax1.tick_params(axis='both', which='both', direction='in', bottom='on', right='on')

ax2.plot(x, kx, 'k', linewidth=1.0)
ax2.set_xlim(xmin,xmax)
ax2.set_xlabel(r'Source Height /$r_g$')
ax2.set_ylabel('Density', rotation='270', labelpad=20.0)
ax2.yaxis.set_label_position('right')
ax2.yaxis.set_ticks_position('right')
ax2.xaxis.set_label_position('top')
ax2.xaxis.set_ticks_position('top')
ax2.tick_params(axis='both', which='both', direction='in', bottom='on', left='on')

# # ax3.set_xticklabels([]) ; ax3.set_yticklabels([])
# # Contourf plot
cfset = ax3.contourf(xx, yy, f, cmap=cmcust) #cmap=plt.cm.get_cmap(scheme))
# # cbar4 = plt.colorbar(cfset, pad=0.05)#, ticks=[-0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
# # cbar4.ax.set_ylabel('Density', rotation='270', labelpad=25.0)
# # Contour plot
cset = ax3.contour(xx, yy, f, colors='k', linewidths=0.5)
# # Hist2d plot
# cfset = ax3.hist2d(x,y[::-1],bins=[56,50], cmap=cmcust)


# Label plot
ax3.invert_xaxis()
ax3.set_xlim(xmin, xmax)
ax3.set_ylim(ymin, ymax)
ax3.set_xlabel(r'Source Height /$r_g$')
ax3.set_ylabel(r'Source Velocity /$c$', rotation='270', labelpad=20.0)
ax3.yaxis.set_label_position('right')
ax3.yaxis.set_ticks_position('right')
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
    if (j==0):
        M_bh = pow(10.0, 7.30)*M_sun ; name = 'Negrete et al. (2012)'
        r_g0 = (G*M_bh)/(c**2.0)
    if (j==1):
        M_bh = pow(10.0, 7.30+0.23)*M_sun ; name = 'Mass + error'
    if (j==2):
        M_bh = pow(10.0, 7.30-0.19)*M_sun ; name = 'Mass -- error'

    ## III Zw 2
    # if (j==0):
    #     M_bh = pow(10.0, 8.03)*M_sun ; name = 'van den Bosch (2016)'
    #     r_g0 = (G*M_bh)/(c**2.0)
    # if (j==1):
    #     M_bh = pow(10.0, 8.03+0.26)*M_sun ; name = '+'
    # if (j==2):
    #     M_bh = pow(10.0, 8.03-0.26)*M_sun ; name = '--'

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

kxmax_index, kxmax_value = max(enumerate(kx), key=operator.itemgetter(1))
kymax_index, kymax_value = max(enumerate(ky), key=operator.itemgetter(1))

ax1.axhline(y=y[kymax_index], color='k', dashes=[5,3], linewidth=1.0)
ax2.axvline(x=x[kxmax_index], color='k', dashes=[5,3], linewidth=1.0)
ax3.axhline(y=y[kymax_index], color='k', dashes=[5,3], linewidth=1.0)
ax3.axvline(x=x[kxmax_index], color='k', dashes=[5,3], linewidth=1.0)
ax3.scatter(x[kxmax_index], y[kymax_index], c='r', s=15.0)

print "Plotting is done!"
print "1D velocity: beta = ", y[kymax_index]
print "1D height:      z = ", x[kxmax_index]
print ""
# print "2D KDE:         t = ", t1-t0
# print "1D sorting:     t = ", t3-t2
# print "1D x KDE:       t = ", t5-t4
# print "1D y KDE:       t = ", t7-t6

# np.savetxt('2d_kde.txt', f)
# np.savetxt('multi_1d_xkde.txt', kx)
# np.savetxt('multi_1d_ykde.txt', ky)

# plt.savefig('/Users/agonzalez/Desktop/IZw1_kde_separate_gallifrey.png', bbox_inches='tight', dpi=300)
# plt.savefig('/Users/agonzalez/Desktop/IIIZw2_kde_separate_xmm.png', bbox_inches='tight', dpi=300)
# plt.savefig('/Users/agonzalez/Desktop/contour_place_holder_colorbar.ps', format='ps', bbox_inches='tight', dpi=300)

plt.show()
