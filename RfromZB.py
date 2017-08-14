# -*- coding: utf-8 -*-
"""
@author: adamg
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.stats import gaussian_kde

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally
plt.rc('font',family='serif')
plt.rc('text',usetex=True)

####################################################################################################
# Compute the approximation for R given z and beta
def R_calc(h,v):
    a = 0.998

    mu_in = (2.0 - h**2.0)/(h**2.0 + a**2.0)
    mu_out = (2.0*h)/(h**2.0 + a**2.0)
    mu_in_p = (mu_in - v)/(1.0 - v*mu_in)
    mu_out_p = (mu_out - v)/(1.0 - v*mu_out)

    value = (mu_out_p - mu_in_p)/(1.0 - mu_out_p)

    return value
####################################################################################################

res = 500
z = np.logspace(np.log10(2.0), np.log10(20.0), res)
# z = np.linspace(2.0, 20.0, res)
beta = np.linspace(0.0, 1.0, res)
Rvs = np.zeros([res+1,res+1])

# compute R as function of source height and source velocity
for i in range (0, res):
    for j in range (0, res):
        Rvs[0,j+1] = beta[j]
        Rvs[i+1,j+1] = R_calc(z[i],beta[j])
    Rvs[i+1,0] = z[i]


# # plot up R vs Z and B
# plt.figure(1)
# ax1 = plt.subplot(211)
# ax2 = plt.subplot(212)
#
# for k in range (0,res):
#     ax1.errorbar(x=beta, y=Rvs[k+1,1:], fmt='-o', markersize=3.0, linewidth=1.0)
#     ax2.errorbar(x=z, y=Rvs[1:,k+1], fmt='-o', markersize=3.0, linewidth=1.0)
#
# ax1.axhline(y=0.54, color='k', linewidth=1.0)
# ax1.fill_between(x=beta, y1=0.54-0.04, y2=0.54+0.04, color='k', alpha=0.25)
# ax1.set_xlim(beta[0],beta[-1])
# ax1.set_ylim(0.0,1.0)
# ax1.set_xlabel(r"Velocity, $\beta$")
# ax1.set_ylabel(r"Refl. Frac., $R$")
# # plt.legend(loc='best', ncol=1, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)
# # ax1.get_legend()
#
# ax2.axhline(y=0.54, color='k', linewidth=1.0)
# ax2.fill_between(x=z, y1=0.54-0.04, y2=0.54+0.04, color='k', alpha=0.25)
# ax2.set_xlim(z[0],z[-1])
# ax2.set_ylim(beta[0],beta[-1])
# ax2.set_xlabel(r"Height, $z$")
# ax2.set_ylabel(r"Refl. Frac., $R$")
# # plt.legend(loc='best', ncol=1, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)
# # ax2.get_legend()
#
# # plt.savefig('/Users/agonzalez/Desktop/IZw1_vesc-h.png', bbox_inches='tight', dpi=300)
# plt.show()


# Compute and plot the pairs (z,b) that match the reflection fraction desired
c = 0
pairs = [[0,0]]
for i in range (0, res):
    for j in range (0, res):
        if (Rvs[i+1,j+1]<=(0.54+0.04)) and (Rvs[i+1,j+1]>=(0.54-0.04)):
            c += 1
            pairs = np.append(pairs,[[Rvs[i+1,0],Rvs[0,j+1]]], axis=0)

# Calculate the point density
# xy = np.vstack([pairs[1:,0],pairs[1:,1]])
# z = gaussian_kde(xy)(xy)

print 'Number of sources within R = 0.54pm0.04 =', c
print ''

plt.figure(2)
# plt.scatter(x=pairs[1:,0], y=pairs[1:,1], c='r', s=50.0)
plt.hist2d(pairs[1:,0], pairs[1:,1], (25,25), cmap=plt.cm.plasma) ; plt.colorbar()
plt.xlabel(r'Height, $z$')
plt.ylabel(r'Velocity, $\beta$')
plt.xlim(z[0]-0.5,z[-1]+0.5)
plt.ylim(0.34,0.90)
# plt.savefig('/Users/agonzalez/Desktop/IZw1_VH_parameterspace.png', bbox_inches='tight', dpi=300)
plt.show()
