#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 13:21:31 2017

@author: agonzalez
"""

### New covariance calculator for I Zw 1

import os
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1
plt.rc('font',family='serif')


####################################################################################################
# Compute the avg count rate of a file and get rid of any NaN counts
def data_cleaner(d_t_raw, d_r_raw, d_e_raw):
    n = 0
    for i in range (0,len(d_t_raw)):
        if (math.isnan(d_r_raw[i]) == False):
            n += 1
    d_t = np.zeros(n)
    d_r = np.zeros(n)
    d_e = np.zeros(n)
    n = 0
    for i in range (0,len(d_t_raw)):
        if (math.isnan(d_r_raw[i]) == False):
            d_t[n] = d_t_raw[i]
            d_r[n] = d_r_raw[i]
            d_e[n] = d_e_raw[i]
            n += 1
    d_t = d_t - d_t[0]
    a_r = np.average(d_r)

    return d_t, d_r, d_e, n, a_r
####################################################################################################

####################################################################################################
# Open the fits file that contains the light curve
def fits_open(filename):
    fitsfile = fits.open(filename)
    data = fitsfile[1].data
    fitsfile.close()
    t_raw = data.field('TIME') ; t_raw = t_raw - t_raw[0]
    r_raw = data.field('RATE')
    e_raw = data.field('ERROR')

    return t_raw, r_raw, e_raw
####################################################################################################

####################################################################################################
# Compute the PSD
def psd_calc(time, rate):
    # performing the DFT
    n_bins = len(time)
    k = np.arange(n_bins-1)
    freq = k/max(time)
    DFT = np.fft.fft(rate)
    t_bins = time[:-1]
    dt = t_bins[1] - t_bins[0]

    # grabbing only the relevant parts of frq and DFT
    half_n_bins = int((n_bins-1.0)/2.0)
    freq = freq[range(half_n_bins)]
    DFT = DFT[range(half_n_bins)]
    dfreq = freq[1] - freq[0]

    # computing the PSD and background level
    PSD = (2.0*dt*abs(DFT)**2.0)/(n_bins*avg_rate**2.0)

    return freq, dfreq, PSD
####################################################################################################

####################################################################################################
# Binner
def binner(bins, x_data, y_data):
    n = len(bins)-1
    bin_counts, binned_y_data, ssq, s = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    for j in range (0, len(y_data)):
        for i in range (0, n):
            if (bins[i] <= x_data[j] <= bins[i+1]):
                bin_counts[i] += 1
                binned_y_data[i] += y_data[j]

    binned_y_data = binned_y_data / bin_counts

    for j in range (0, len(y_data)):
        for i in range (0, n):
            if (bins[i] <= x_data[j] <= bins[i+1]):
                ssq[i] += (y_data[j]-binned_y_data[i])**2.0
    ssq = ssq / (bin_counts-1.)
    s = np.sqrt(ssq/bin_counts)
    # s = binned_y_data / np.sqrt(bin_counts)  # I think what Pavg / sqrt(KM) in Uttley et al. 2014 -> turns out to be essentially the same as above

    return binned_y_data, bin_counts, s
####################################################################################################


os.chdir("/Users/agonzalez/Documents/Research/Data/Mrk766/265_May2017")

# open the light curve filename list
with open('265_lccor_list.txt','r') as lcfile:
    lc_fnames = [line.rstrip('\n') for line in lcfile]
lcfile.close()
n_lc = len(lc_fnames)

# open the background filename list
with open('265_bgraw_list.txt','r') as bgfile:
    bg_fnames = [line.rstrip('\n') for line in bgfile]
bgfile.close()
n_bg = len(bg_fnames)

# just a sanity check
if (n_lc == n_bg):
    n_RUNS = n_lc
else:
    n_RUNS = 0
    print "Mismatch bewteen the number of light curves and background files"

# set up all of the final output variables and the number of files to go through
energy = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.90, 1.15, 1.45, 1.8, 2.25, 2.75, 3.25, 3.75, 4.5, 5.5, 6.5, 7.5, 9.0]
energy = energy[:n_RUNS]

Df_LF = pow(10,-4.0) - pow(10,-4.6)
Df_MF = pow(10,-3.5) - pow(10,-4.0)
Df_HF = pow(10,-3.0) - pow(10,-3.5)


for RUN in range (0,n_RUNS):
    print "RUN NUMBER: ", RUN+1

    # read in the source and background lightcurves
    lc_t_raw, lc_r_raw, lc_e_raw = fits_open(lc_fnames[RUN])
    bg_t_raw, bg_r_raw, bg_e_raw = fits_open(bg_fnames[RUN])

    # getting rid of last 15ks
    lc_t_raw = lc_t_raw[:1051] ; lc_r_raw = lc_r_raw[:1051] ; lc_e_raw = lc_e_raw[:1051]
    bg_t_raw = bg_t_raw[:1051] ; bg_r_raw = bg_r_raw[:1051] ; bg_e_raw = bg_e_raw[:1051]
    lc_r_raw = lc_r_raw - min(lc_r_raw)

    # remove any NaN counts and get average count rate
    lc_t, lc_r, lc_e, idx, avg_rate = data_cleaner(lc_t_raw, lc_r_raw, lc_e_raw)
    bg_t, bg_r, bg_e, bg_idx, avg_bg_rate = data_cleaner(bg_t_raw, bg_r_raw, bg_e_raw)

    # fix this particular data point
    lc_r[906] = (lc_r[905] + lc_r[907]) / 2.0
    lc_e[906] = (lc_e[905] + lc_e[907]) / 2.0

    # Plot the light curve
    # plt.figure(1)
    # plt.fill_between(x=lc_t, y1=lc_r-lc_e, y2=lc_r+lc_e, color='r', alpha=0.75)
    # plt.fill_between(x=bg_t, y1=bg_r-bg_e, y2=bg_r+bg_e, color='b', alpha=0.75)
    # plt.xlabel('Time') ; plt.ylabel('Counts / sec')
    # plt.xlim(min(lc_t),max(lc_t))
    # plt.show()

    # compute the PSD
    frq, df, PSD = psd_calc(lc_t, lc_r)

    # bin up the PSD
    n_bins = 10
    frq_bins = np.logspace(np.log10(frq[1]), np.log10(frq[-1]), n_bins)
    PSD_binned, PSD_binned_counts, PSD_binned_error = binner(frq_bins, frq, PSD)
    avg_frq_bins = np.zeros(n_bins-1)
    avg_frq_bins_err = np.zeros(n_bins-1)
    for i in range (0,n_bins-1):
        avg_frq_bins[i] = (frq_bins[i+1]+frq_bins[i])/2.
        avg_frq_bins_err[i] = frq_bins[i+1] - avg_frq_bins[i]

    # Plot the raw and binned up PSD
    plt.figure(2)
    plt.xscale('log') ; plt.yscale('log')
    plt.scatter(x=frq[1:], y=PSD[1:], c='r', marker='o', alpha=0.75)
    plt.axvline(x=frq[-1], color='k', dashes=[5,3], linewidth=1.0)
    plt.errorbar(x=avg_frq_bins, y=PSD_binned, yerr=PSD_binned_error, fmt='o', markersize=5, color='k', ecolor='k', capthick=0, linewidth=1.0, alpha=1.0)
    plt.step(x=avg_frq_bins, y=PSD_binned, color='k', linewidth=1.0, where='mid')
    plt.xlabel('Frequency (Hz)') ; plt.ylabel('Power (Hz$^{-1}$)')
    # plt.savefig('/Users/agonzalez/Desktop/step_plot.png', bbox_inches='tight', dpi=300)
    plt.show()


# #    plottertron(lc_t, lc_r, lc_e, 'Time (s)', 'Count Rate (ct/s)', 0, 'b', '265')
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plottertron(frq[1:], PSD[1:], np.zeros(1), 'Frequency (Hz)', 'Power (Hz$^{-1}$)', 1, '0.65', '265')
#     plottertron(frq_bins, PSD_binned, np.zeros(1), 'Frequency (Hz)', 'Power (Hz$^{-1}$)', 1, 'r', '265')
#
#     print "Mean Energy = ", energy[RUN]
#     print "Average source rate = ", avg_rate
#     print "Average background rate = ", avg_bg_rate
#
#     print ""
#
#
#

#    # compute Poisson noise level
#    PN_lev = 2.0*(avg_rate - avg_bg_rate)/(avg_rate**2.0)
#    count_LF = 0 ; count_MF = 0 ; count_HF = 0
#
#    # computing the excess variance total and for each frequency band
#    for i in range (0,len(frq)):
#        if (pow(10,-4.6) <= frq[i] <= pow(10,-4.0)):
#            sig_NXS_LF[RUN] += (PSD[i]-PN_lev)*df
#            count_LF += 1.0
#        if (pow(10,-4.0) <= frq[i] <= pow(10,-3.5)):
#            sig_NXS_MF[RUN] += (PSD[i]-PN_lev)*df
#            count_MF += 1.0
#        if (pow(10,-3.5) <= frq[i] <= pow(10,-3.0)):
#            sig_NXS_HF[RUN] += (PSD[i]-PN_lev)*df
#            count_HF += 1.0
#
#    sig_NXS_LF_ERR[RUN] = np.sqrt( ((PN_lev*Df_LF)**2.0 + 2.0*PN_lev*Df_LF*sig_NXS_LF[RUN]) / count_LF )
#    sig_NXS_MF_ERR[RUN] = np.sqrt( ((PN_lev*Df_MF)**2.0 + 2.0*PN_lev*Df_MF*sig_NXS_MF[RUN]) / count_MF )
#    sig_NXS_HF_ERR[RUN] = np.sqrt( ((PN_lev*Df_HF)**2.0 + 2.0*PN_lev*Df_HF*sig_NXS_HF[RUN]) / count_HF )
#
#    sig_RMS_LF[RUN] = avg_rate*np.sqrt(sig_NXS_LF[RUN])
#    sig_RMS_MF[RUN] = avg_rate*np.sqrt(sig_NXS_MF[RUN])
#    sig_RMS_HF[RUN] = avg_rate*np.sqrt(sig_NXS_HF[RUN])
#
#    sig_RMS_LF_ERR[RUN] = np.sqrt( sig_NXS_LF[RUN] + sig_NXS_LF_ERR[RUN] ) - sig_RMS_LF[RUN]
#    sig_RMS_MF_ERR[RUN] = np.sqrt( sig_NXS_MF[RUN] + sig_NXS_MF_ERR[RUN] ) - sig_RMS_MF[RUN]
#    sig_RMS_HF_ERR[RUN] = np.sqrt( sig_NXS_HF[RUN] + sig_NXS_HF_ERR[RUN] ) - sig_RMS_HF[RUN]

#plt.subplot(212)
#frq_LF = [pow(10,-4.6),pow(10,-4.0)]
#frq_MF = [pow(10,-4.0),pow(10,-3.5)]
#frq_HF = [pow(10,-3.5),pow(10,-3.0)]
#plt.fill_between(frq_LF, 1e-6,1e+6, facecolor='red', alpha=0.25, interpolate=True)
#plt.fill_between(frq_MF, 1e-6,1e+6, facecolor='green', alpha=0.25, interpolate=True)
#plt.fill_between(frq_HF, 1e-6,1e+6, facecolor='blue', alpha=0.25, interpolate=True)
#
###plt.savefig('../Data/Mrk766/0304030101/999_lc-PSD_all.png',bbox_inches='tight')
#
#
#plt.figure(2)
##plt.subplot(211)
###plt.loglog(energy, sig_NXS, '-sk', label = "Total", linewidth = 2)
##plt.loglog(energy, sig_NXS_LF, ':or', label = "LF", markersize = 5, linewidth = 2)
##plt.loglog(energy, sig_NXS_MF, '-og', label = "MF", markersize = 5, linewidth = 2)
##plt.loglog(energy, sig_NXS_HF, '--ob', label = "HF", markersize = 5, linewidth = 2)
#
#ax = plt.subplot(111)
##ax = plt.subplot(111)
#ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')
#plt.errorbar(energy, sig_NXS_LF, yerr = sig_NXS_LF_ERR, dashes=[3,5], color='r', linewidth=1, ecolor='r', elinewidth=2, capthick=2, label = "LF")
#plt.errorbar(energy, sig_NXS_MF, yerr = sig_NXS_MF_ERR, dashes=[1,0.1], color='g', linewidth=1, ecolor='g', elinewidth=2, capthick=2, label = "MF")
#plt.errorbar(energy, sig_NXS_HF, yerr = sig_NXS_HF_ERR, dashes=[10,5], color='b', linewidth=1, ecolor='b', elinewidth=2, capthick=2, label = "HF")
#
#plt.xlabel("Energy [keV]")
#plt.ylabel("Normalised Excess Variance, $\sigma_{\mathrm{NXS}}^{2}$")
#plt.xlim(0.20,10.0)
##plt.ylim(5e-4, 0.035)
#plt.legend(loc=4,labelspacing=0.1,fontsize=16)
#plt.show()
###plt.savefig('../Data/Mrk766/0304030101/999_NXSvE_werr.png',bbox_inches='tight')

## Fourier-resolved spectra
#ax = plt.subplot(212)
##ax = plt.subplot(111)
#ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')
#plt.errorbar(energy, sig_RMS_LF, yerr = sig_RMS_LF_ERR, dashes=[3,5], color='r', linewidth=1, ecolor='r', elinewidth=2, capthick=2, label = "LF")
#plt.errorbar(energy, sig_RMS_MF, yerr = sig_RMS_MF_ERR, dashes=[1,0.1], color='g', linewidth=1, ecolor='g', elinewidth=2, capthick=2, label = "MF")
#plt.errorbar(energy, sig_RMS_HF, yerr = sig_RMS_HF_ERR, dashes=[10,5], color='b', linewidth=1, ecolor='b', elinewidth=2, capthick=2, label = "HF")
#
#plt.xlabel("Energy [keV]")
#plt.ylabel("RMS Spectra, $\sigma_{\mathrm{rms}}$")
#plt.xlim(0.20,10.0)
#plt.ylim(5e-4, 2.0)
#plt.legend(loc=4,labelspacing=0.1,fontsize=16)
#plt.show()
