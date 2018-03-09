#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: adamg
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import scipy.stats as st
import time
import os

from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally
plt.rc('font',family='serif')
plt.rc('text',usetex=True)


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
cdict2 = {'red':   ((0.00, 1.0, 1.0),
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

         'blue':   ((0.00, 1.0, 1.0),
                    (0.10, 1.0, 1.0),
                    (0.20, 0.0, 0.0),
                    (0.40, 0.0, 0.0),
                    (0.60, 0.0, 0.0),
                    (0.80, 0.0, 0.0),
                    (1.00, 0.0, 0.0)),
        }
cmcust = LinearSegmentedColormap('customcmap', cdict1)


# os.chdir("/Users/agonzalez/Documents/Research/Data/IZw1")
# # data = np.genfromtxt("single_sim.txt")
# data = np.genfromtxt("multi_sim.txt")
# # data = np.genfromtxt("big_sim.txt")
# # data = np.genfromtxt("big_sim_aug16.txt")

os.chdir("/Users/agonzalez/Documents/Research/Data/Mrk1501/VHdet")
# data = np.genfromtxt("suz.txt")
# data = np.genfromtxt("xmm.txt")
# data = np.genfromtxt('xmm_big.txt')

data = np.zeros(shape=(1,2))
for i in range (0, 5):
    RUN = i+1
    temp = np.genfromtxt('xmm_{0:01d}.txt'.format(int(RUN)))
    data = np.append(data, temp, axis=0)

# print data

x, y = data[:,0], data[:,1]
minh, maxh = 2.0, 32.0
minv, maxv = 0.5, 1.0

# print 'Avg height = ', np.average(x)
# print 'Avg velocity = ', np.average(y)

plt.figure()
ax = plt.subplot(111)
##---------------------------------------------------------------------------------------
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
        ax.plot(R[j][:],Vesc[j][:], color=col[j], dashes=[5,3], alpha=0.75, label=name)
    elif (j==0):
        ax.plot(R[j][:],Vesc[j][:], color=col[j], alpha=0.75, label=name)

for i in range (0,res):
    R[3][i] = abs(R[0][i]-R[1][i])
    R[4][i] = abs(R[0][i]-R[2][i])

# ax.fill_betweenx(y=Vesc[0][:], x1=R[0][:]-R[4][:], x2=R[0][:]+R[3][:], facecolor='red', alpha=0.05)
ax.tick_params(axis='both', which='both', direction='in', top='on', right='on')
##---------------------------------------------------------------------------------------

##---------------------------------------------------------------------------------------
xmin, xmax = minh, maxh
ymin, ymax = minv, maxv


# ## Peform the kernel density estimate
# xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# positions = np.vstack([xx.ravel(), yy.ravel()])
# values = np.vstack([x, y])
# kernel = st.gaussian_kde(values)
# f = np.reshape(kernel(positions).T, xx.shape)
# ## Contourf plot
# cfset = ax.contourf(xx, yy, f, cmap=cmcust) #cmap=plt.cm.get_cmap(scheme))
# #cbar4 = plt.colorbar(cfset)
# ## Colour bar
# cbar4 = plt.colorbar(cfset, pad=0.05)#, ticks=[-0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
# cbar4.ax.set_ylabel('Density', rotation='270', labelpad=25.0)
# ## Contour plot
# cset = ax.contour(xx, yy, f, colors='k', linewidths=0.5)


## 2D histogram
plt.hist2d(x,y, bins=[30,25], range=[[xmin,xmax],[ymin,ymax]], cmap=cmcust)
# plt.hexbin(x,y, gridsize=50, cmap=cmcust)
cb = plt.colorbar()

# ## Doing individual point density
# xy = np.vstack([x,y])
# z = gaussian_kde(xy)(xy)
# idx = z.argsort()
# x, y, z = x[idx], y[idx], z[idx]
# ## Plotting
# # fig, ax = plt.subplots()
# ax.scatter(x, y, c=z, s=50, edgecolor='')
# # plt.show()


ax.set_xlim(2, 30)
ax.set_ylim(ymin, ymax)

# Label plot
#ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel(r'Source Height /$r_g$')
ax.set_ylabel(r'Source Velocity /$c$')
##---------------------------------------------------------------------------------------

# plt.savefig('hist2dpairs.png', bbox_inches='tight', dpi=300)
plt.show()
