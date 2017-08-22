#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: agonzalez
"""

### New covariance calculator for I Zw 1

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1
plt.rc('font',family='serif')
plt.rc('text',usetex=True)

####################################################################################################
# Compute the escape velocity for a black hole of mass M at a height R above the black hole
def vesc_calc(G,M,R,c):
    v = np.sqrt((2.0*G*M)/R)/c

    return v
####################################################################################################

G = 6.674e-11
c = 2.998e8
M_sun = 1.989e30

plt.figure()
ax = plt.subplot(111)
col = ['k','b','r']

res = 25
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

    print "Mass of I Zw 1 BH [kg]   = ", M_bh
    print "Schwarzschild radius [m] = ", R_s
    print "Gravitationl radius [m]  = ", r_g
    R[j][:] = R[j][:]/r_g0

    ax.semilogx(R[j][:],Vesc[j][:], color=col[j], alpha=0.75, label=name)

for i in range (0,res):
    R[3][i] = abs(R[0][i]-R[1][i])
    R[4][i] = abs(R[0][i]-R[2][i])

ax.fill_betweenx(y=Vesc[0][:], x1=R[0][:]-R[4][:], x2=R[0][:]+R[3][:], color='k', alpha=0.15)
# ax.scatter(x=10.0, y=0.45, s=25.0, c='g', label=r'10$r_g$, 0.45')
# ax.scatter(x=18.0, y=0.39, s=25.0, c='c', label=r'18$r_g$, 0.39')
# ax.scatter(x=27.0, y=0.36, s=25.0, c='m', label=r'27$r_g$, 0.36')
plt.legend(loc=1, ncol=1, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)
ax.axhline(y=0.1, color='k', dashes=[5,3])
ax.axvline(x=200.0, color='k', dashes=[5,3])
ax.get_legend()
ax.tick_params(axis='both', which='both', direction='in', top='on', right='on')
ax.set_xlabel('Source Height /$r_{g,0}$')
ax.set_ylabel('Escape Velocity /$c$')
ax.set_xticks([1,10,100,1000])
ax.set_xticklabels([1,10,100,1000])
# ax.set_yticks([1e-1,1.0])
# ax.set_yticklabels([0.1,1.0])
ax.set_ylim(0.0,1.0)

# plt.savefig('/Users/agonzalez/Desktop/IZw1_vesc-h.png', bbox_inches='tight', dpi=300)
plt.show()
