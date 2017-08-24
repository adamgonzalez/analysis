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
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally
plt.rc('font',family='serif')
plt.rc('text',usetex=True)

# data = np.genfromtxt("single_sim.txt")
# data = np.genfromtxt("multi_sim.txt")
# data = np.genfromtxt("big_sim.txt")
data = np.genfromtxt("big_sim_aug16.txt")
x, y = data[:,0], data[:,1]
minh, maxh = 2.0, 30.0
scheme = 'Greys'

plt.figure()
ax = plt.subplot(111)
####################################################################################################
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
    if (j==0):
        M_bh = pow(10.0, 7.30)*M_sun ; name = 'Negrete et al. (2012)'
        r_g0 = (G*M_bh)/(c**2.0)
    if (j==1):
        M_bh = pow(10.0, 7.30+0.23)*M_sun ; name = 'Mass + error'
    if (j==2):
        M_bh = pow(10.0, 7.30-0.19)*M_sun ; name = 'Mass -- error'

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

ax.fill_betweenx(y=Vesc[0][:], x1=R[0][:]-R[4][:], x2=R[0][:]+R[3][:], facecolor='red', alpha=0.05)
# ax.scatter(x=11.0, y=0.45, s=35.0, c='k', label=r'11$r_g$, 0.45')
# ax.scatter(x=15.0, y=0.41, s=35.0, c='k', label=r'27$r_g$, 0.36')
# ax.scatter(x=18.0, y=0.39, s=35.0, c='k', label=r'18$r_g$, 0.39')
# ax.scatter(x=27.0, y=0.36, s=35.0, c='k', label=r'18$r_g$, 0.39')
# plt.legend(loc=1, ncol=1, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)
# ax.get_legend()
ax.tick_params(axis='both', which='both', direction='in', top='on', right='on')
# ax.set_xlabel('Source Height /$r_{g,0}$')
# ax.set_ylabel('Escape Velocity /$c$')
# ax.set_xticks([1,10,100,1000])
# ax.set_xticklabels([1,10,100,1000])
# ax.set_yticks([1e-1,1.0])
# ax.set_yticklabels([0.1,1.0])
# ax.set_ylim(0.0,1.0)
####################################################################################################

##---------------------------------------------------------------------------------------
# # Do it the Seaborn way
# data = np.random.multivariate_normal((0, 0), [[0.8, 0.05], [0.05, 0.7]], 100)
# x = data[:, 0]
# y = data[:, 1]
xmin, xmax = minh, maxh
ymin, ymax = 0.25, 0.75

# Peform the kernel density estimate
t0 = time.time()
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
t1 = time.time()
print "Grid mesh: ", t1-t0

t2 = time.time()
positions = np.vstack([xx.ravel(), yy.ravel()])
t3 = time.time()
print "Positions: ", t3-t2

t4 = time.time()
values = np.vstack([x, y])
t5 = time.time()
print "Values:    ", t5-t4

t6 = time.time()
kernel = st.gaussian_kde(values)
t7 = time.time()
print "Kernel:    ", t7-t6

t8 = time.time()
f = np.reshape(kernel(positions).T, xx.shape)
# f = np.genfromtxt("multi_sim_fvals.txt")
t9 = time.time()
print "Reshape:   ", t9-t8
print f.shape

# filename = open("multi_sim_fvals.txt","w")
# np.savetxt(filename, f)


# cdict1 = {'red':   ((0.00, 1.0, 1.0),
#                     (0.10, 1.0, 1.0),
#                     (0.25, 0.66, 0.66),
#                     (0.75, 0.33, 0.33),
#                     (1.00, 0.0, 0.0)),
#          'green':  ((0.00, 1.0, 1.0),
#                     (0.10, 1.0, 1.0),
#                     (0.25, 0.66, 0.66),
#                     (0.75, 0.33, 0.33),
#                     (1.00, 0.0, 0.0)),
#          'blue':   ((0.00, 1.0, 1.0),
#                     (0.10, 1.0, 1.0),
#                     (0.25, 0.66, 0.66),
#                     (0.75, 0.33, 0.33),
#                     (1.00, 0.0, 0.0)),
#         }
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


# fig = plt.figure()
# ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
# Contourf plot
cfset = ax.contourf(xx, yy, f, cmap=cmcust) #cmap=plt.cm.get_cmap(scheme))
#cbar4 = plt.colorbar(cfset)
cbar4 = plt.colorbar(cfset, pad=0.05)#, ticks=[-0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
cbar4.ax.set_ylabel('Probability Density', rotation='270', labelpad=25.0)


# Or kernel density estimate plot instead of the contourf plot
# ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
# Contour plot
cset = ax.contour(xx, yy, f, colors='k', linewidths=0.5)
# Label plot
#ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel(r'Source Height /$r_g$')
ax.set_ylabel(r'Source Velocity /$c$')
##---------------------------------------------------------------------------------------

##---------------------------------------------------------------------------------------
# # Do it the hist2d way
# plt.figure()
# plt.subplot(221)
# plt.hist2d(x, y, (10,10), cmap=plt.get_cmap(scheme)) ; plt.colorbar()
# plt.title(r'$10\times10$')
# plt.xlabel(r'Height, $z$') ; plt.ylabel(r'Velocity, $\beta$') ; plt.xlim(minh,maxh) ; plt.ylim(0.0,1.0) ; plt.ylim(0.34,0.9)
#
# plt.subplot(222)
# plt.hist2d(x, y, (25,25), cmap=plt.get_cmap(scheme)) ; plt.colorbar()
# plt.title(r'$25\times25$')
# plt.xlabel(r'Height, $z$') ; plt.ylabel(r'Velocity, $\beta$') ; plt.xlim(minh,maxh) ; plt.ylim(0.0,1.0) ; plt.ylim(0.34,0.9)
#
# plt.subplot(223)
# plt.hist2d(x, y, (50,50), cmap=plt.get_cmap(scheme)) ; plt.colorbar()
# plt.title(r'$50\times50$')
# plt.xlabel(r'Height, $z$') ; plt.ylabel(r'Velocity, $\beta$') ; plt.xlim(minh,maxh) ; plt.ylim(0.0,1.0) ; plt.ylim(0.27,0.9)
#
# plt.subplot(224)
# plt.hist2d(x, y, (100,100), cmap=plt.get_cmap(scheme)) ; plt.colorbar()
# plt.title(r'$100\times100$')
# plt.xlabel(r'Height, $z$') ; plt.ylabel(r'Velocity, $\beta$') ; plt.xlim(minh,maxh) ; plt.ylim(0.0,1.0) ; plt.ylim(0.34,0.9)


# ax.hist2d(x, y, (50,50), cmap=plt.get_cmap(scheme)) #; plt.colorbar()
# ax.set_xlabel(r'Height, $z$') ; ax.set_ylabel(r'Velocity, $\beta$') ; ax.set_xlim(minh,maxh) ; ax.set_ylim(0.25,0.75)
##---------------------------------------------------------------------------------------





# plt.savefig('/Users/agonzalez/Desktop/single_sim.png', bbox_inches='tight', dpi=300)
# plt.savefig('/Users/agonzalez/Desktop/multi_sim.png', bbox_inches='tight', dpi=300)
# plt.savefig('/Users/agonzalez/Desktop/big_sim_grey.png', bbox_inches='tight', dpi=300)
# plt.savefig('/Users/agonzalez/Desktop/big_sim_aug18_grey.png', bbox_inches='tight', dpi=300)
# plt.savefig('/Users/agonzalez/Desktop/place_holder.ps', format='ps', bbox_inches='tight', dpi=300)
plt.savefig('/Users/agonzalez/Desktop/contour_place_holder_colorbar.ps', format='ps', bbox_inches='tight', dpi=300)
plt.show()
