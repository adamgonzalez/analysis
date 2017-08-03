#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:57:02 2017

@author: agonzalez
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import gridspec
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally


####################################################################################################
# read in the qdp data files
def qdp_read(filename, NO):
    data = np.genfromtxt(filename)

    NO = NO-2
    NO = int(NO)
    dat_x = data[:NO,0]
    dat_x_error = data[:NO,1]
    dat_y = data[:NO,2]
    dat_y_error = data[:NO,3]

    mo = data[:NO,4]

    NO = NO+2
    NO = int(NO)
    res_x = data[NO:,0]
    res_x_error = data[NO:,1]
    res_y = data[NO:,2]
    res_y_error = data[NO:,3]

    return dat_x, dat_x_error, dat_y, dat_y_error, mo, res_x, res_x_error, res_y, res_y_error
####################################################################################################

####################################################################################################
# read in the qdp data files
def asca_read(filename, NO):
    data = np.genfromtxt(filename)

    NO = NO-2
    NO = int(NO)
    dat_x = data[:NO,0]
    dat_x_error = data[:NO,1]
    dat_y = data[:NO,2]
    dat_y_error = data[:NO,3]

    mo = data[:NO,4]

    NO = NO+2
    NO = int(NO)
    res_x = data[NO:,0]
    res_x_error = data[NO:,1]
    res_y = data[NO:,2]
    res_y_error = data[NO:,3]

    return dat_x, dat_x_error, dat_y, dat_y_error, mo, res_x, res_x_error, res_y, res_y_error
####################################################################################################

####################################################################################################
# Plot x and y with labels (trying to do it)
def plottertron(x, x_error, y, y_error, line_color, line_label, xscale, yscale):
    ax.set_xlabel('Energy ($\mathrm{keV}$)')
    ax.set_ylabel('Normalized Counts ($\mathrm{s^{-1}}$ $\mathrm{keV^{-1}}$)')
    if (xscale == 'log'):
        ax.set_xscale('log')
    if (yscale == 'log'):
        ax.set_yscale('log')
    ax.errorbar(x, y, xerr=x_error, yerr=y_error, ecolor='0.35', capthick=1, color=line_color, label=line_label, fmt='o', markersize=0)
    ax.legend(loc='best', labelspacing=0.1, fontsize=16)
    plt.show()
####################################################################################################


fig = plt.figure()

# ASCA Data
#e1, e1_err, f1, f1_err, m1, r1_e, r1_e_err, r1, r1_err, e2, e2_err, f2, f2_err, m2, r2_e, r2_e_err, r2, r2_err = asca_read('asca_sis_tbabspo.qdp',92)
#e1, e1_err, f1, f1_err, m1, r1_e, r1_e_err, r1, r1_err, e2, e2_err, f2, f2_err, m2, r2_e, r2_e_err, r2, r2_err = asca_read('asca_sis_tbabspo_zgauss_bb.qdp',92)

# XMM Data
energy, energy_err, flux, flux_err, model, res_e, res_e_err, res, res_err = qdp_read('xmm_tbabspo.qdp',92)
#energy, energy_err, flux, flux_err, model, res_e, res_e_err, res, res_err = qdp_read('xmm_tbabspo_zgauss_bb.qdp',92)

# SWIFT Data
#energy, energy_err, flux, flux_err, model, res_e, res_e_err, res, res_err = qdp_read('swift_tbabspo.qdp',206)
#energy, energy_err, flux, flux_err, model, res_e, res_e_err, res, res_err = qdp_read('swift_tbabspo_zgauss_bb.qdp',206)

# SUZAKU Data
#energy, energy_err, flux, flux_err, model, res_e, res_e_err, res, res_err = qdp_read('suzaku_tbabspo.qdp',233)
#energy, energy_err, flux, flux_err, model, res_e, res_e_err, res, res_err = qdp_read('suzaku_tbabspo_zgauss_bb.qdp',233)


ax = fig.add_subplot(211)
plottertron(energy, energy_err, flux, flux_err, 'k', 'Data', 'log', 'log')
plottertron(energy, np.zeros(len(energy)), model, np.zeros(len(energy)), 'r', 'Model', 'log', 'log')
plt.grid(which='both',linestyle='-',color='0.6')

ax = fig.add_subplot(212)
ax.axhline(y=1.0, xmin=0.005, xmax=10.0, color='r', linewidth=2)
plottertron(res_e, res_e_err, res, res_err, 'k', 'Residuals', 'log', 'linear')

#plt.savefig('suzaku_tbabspo_zgauss_bb.png',bbox_inches='tight')
