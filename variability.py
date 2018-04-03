#!/usr/bin/env
"""
Author: Adam Gonzalez
Desc. : This script will performa basic light curve analysis on the corrected & background light
        curves provided based on the bin size specified.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
import math
import argparse
import sys
from functions import *
from astropy.io import fits
from matplotlib.ticker import ScalarFormatter
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['axes.linewidth'] = 1.25 #set the value globally
plt.rc('font', family='serif', weight='medium')
# plt.rc('text', usetex=True)


### Read in the source-background and background .lc light curves
parser = argparse.ArgumentParser(description='This script will perform a basic light curve analysis on the corrected & background light curves provided based on the bin size specified. \
                                \n\nNote: When using the -hr flag to compute hardness ratio the script will compute -s/-b where -b will be used as the second light curve instead of as a background.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-hr", "--hardnessratio", help="Compute hardness ratio (type 'y' to compute)", type=str, default='none')
parser.add_argument("-ps", "--powerspectrum", help="Compute the power spectrum (type 'y' to compute)", type=str, default='none')
parser.add_argument("-s", "--srcfilename", help="Corrected source light curve filename", type=str, default='none')
parser.add_argument("-b", "--bkgfilename", help="Background light curve filename", type=str, default='none')
parser.add_argument("-tbin", "--lcbinsize", help="Light curve bin size in seconds", type=float, default=0)
parser.add_argument("-pbin", "--psbinsize", help="Power spectrum number of bins", type=float, default=0)

if len(sys.argv) == 1:
    parser.print_help()
    print help
    sys.exit(1)
args = parser.parse_args()

# c = 0
# for i in range (c,3):
#     for j in range (c,3):
#         print "i=", i
#         print "j=", j
#         print ""
#         c += 1

### Read in the files given by the user ##################################
print "\n*************************"
print "Reading in files..."
lc_file = args.srcfilename
bg_file = args.bkgfilename
bin_size = args.lcbinsize

lcfits = fits.open(lc_file)
lcdata = lcfits[1].data
lcfits.close()
lc_t_raw = lcdata.field('TIME') ; lc_t_raw = lc_t_raw - lc_t_raw[0]
lc_r_raw = lcdata.field('RATE')
lc_e_raw = lcdata.field('ERROR')

lc_t_raw = lc_t_raw[:12500]
lc_r_raw = lc_r_raw[:12500]
lc_e_raw = lc_e_raw[:12500]

bgfits = fits.open(bg_file)
bgdata = bgfits[1].data
bgfits.close()
bg_t_raw = bgdata.field('TIME') ; bg_t_raw = bg_t_raw - bg_t_raw[0]
bg_r_raw = bgdata.field('RATE')
bg_e_raw = bgdata.field('ERROR')

lc_t = lc_t_raw ; lc_r = lc_r_raw ; lc_e = lc_e_raw
bg_t = bg_t_raw ; bg_r = bg_r_raw ; bg_e = bg_e_raw
print "...done!"

### tell them where the NaNs are
for i in range (0, len(lc_t)):
    if (math.isnan(lc_r[i])==True):
        print "NaN count at:     ", lc_t[i]

### tell them where there are zero counts
for i in range (0, len(lc_t)):
    if (lc_r[i] <= 0.0):
        print "Zero count at:    ", lc_t[i]

### tell them where the background dominates
for i in range (0, len(lc_t)):
    if (abs(bg_r[i]/lc_r[i]) >= 0.15):
        print "Bkg/src > 0.15 at: ", lc_t[i]

##########################################################################


### Bin up the light curve with the desired binning ######################
if (bin_size != 0):
    print "\nBinning and cleaning light curves..."
    lc_bin_t, lc_bin_t_err, lc_bin_r, lc_bin_r_err = lcbin(bin_size, lc_t, lc_r, lc_e)
    bg_bin_t, bg_bin_t_err, bg_bin_r, bg_bin_r_err = lcbin(bin_size, bg_t, bg_r, bg_e)
elif (bin_size == 0):
    print "\nUsing raw light curves..."
    lc_bin_t, lc_bin_r, lc_bin_r_err = lc_t, lc_r, lc_e
    lc_bin_t_err = (lc_bin_t[1] - lc_bin_t[0])/2.

    bg_bin_t, bg_bin_r, bg_bin_r_err = bg_t, bg_r, bg_e
    bg_bin_t_err = (bg_bin_t[1] - bg_bin_t[0])/2.

lc_bin_r, lc_bin_r_err, lc_nan_bin_t, lc_nan_bin_t_err, lc_nan_bin_r, lc_nan_bin_r_err = cleanup(lc_bin_t, lc_bin_t_err, lc_bin_r, lc_bin_r_err)
bg_bin_r, bg_bin_r_err, bg_nan_bin_t, bg_nan_bin_t_err, bg_nan_bin_r, bg_nan_bin_r_err = cleanup(bg_bin_t, bg_bin_t_err, bg_bin_r, bg_bin_r_err)

print "...done!"

### Compute the mean, fit to the mean, and fractional variability of that light curves
lc_bin_r_avg, lc_bin_r_avg_err = wmean(lc_bin_r, lc_bin_r_err)
bg_bin_r_avg, bg_bin_r_avg_err = wmean(bg_bin_r, bg_bin_r_err)
lc_bin_chi, lc_bin_redchi = chisqcalc(lc_bin_r, lc_bin_r_err, lc_bin_r_avg)
lc_bin_Fvar, lc_bin_Fvar_err = fvarcalc(lc_bin_r, lc_bin_r_err, lc_bin_r_avg)
print "\nLight curve analysis:"
print "     Average : {0:.3f} +/- {1:.3f} cps".format(lc_bin_r_avg,lc_bin_r_avg_err)
print "     RedChi  : {0:.3f}".format(lc_bin_redchi)
print "     F_var   : {0:.3f} +/- {1:.3f} %".format(lc_bin_Fvar*100.,lc_bin_Fvar_err*100.)
##########################################################################


### Plot the light curve #################################################
# plt.figure(1)
# ax = plt.subplot(111)
# ax.axhline(y=lc_bin_r_avg, color='k', dashes=[5,3], linewidth=1)
# ax.fill_between(x=np.linspace(lc_bin_t[0]-lc_bin_t_err, lc_bin_t[-1]+lc_bin_t_err), y1=lc_bin_r_avg-lc_bin_r_avg_err, y2=lc_bin_r_avg+lc_bin_r_avg_err, facecolor='k', alpha=0.25)
# ax.errorbar(lc_bin_t, lc_bin_r, xerr=lc_bin_t_err, yerr=lc_bin_r_err, color='r', ecolor='r', fmt='o', markersize=0, markerfacecolor='none', linestyle='-', linewidth=1)
# ax.errorbar(lc_nan_bin_t[1:], lc_nan_bin_r[1:], xerr=lc_nan_bin_t_err[1:], yerr=lc_nan_bin_r_err[1:], color='k', ecolor='k', fmt='x', markersize=5, markerfacecolor='none', linewidth=1)
# ax.errorbar(bg_bin_t, bg_bin_r, xerr=bg_bin_t_err, yerr=bg_bin_r_err, color='b', ecolor='b', fmt='o', markersize=0, markerfacecolor='none', linestyle='-', linewidth=1)
# ax.errorbar(bg_nan_bin_t[1:], bg_nan_bin_r[1:], xerr=bg_nan_bin_t_err[1:], yerr=bg_nan_bin_r_err[1:], color='k', ecolor='k', fmt='x', markersize=5, markerfacecolor='none', linewidth=1)
# ax.set_xlabel(r'Time [s]')
# ax.set_ylabel(r'Flux [counts s$^{-1}$]')
# ax.set_xlim(left=lc_bin_t[0]-lc_bin_t_err, right=lc_bin_t[-1]+lc_bin_t_err)
# # ax.set_ylim(bottom=min(lc_bin_r)-max(lc_bin_r_err)*1.1, top=max(lc_bin_r)+max(lc_bin_r_err)*1.1)
# ax.set_ylim(bottom=0., top=max(lc_bin_r)+max(lc_bin_r_err)*1.1)
# ax.tick_params(axis='both', which='both', direction='in', top='on', right='on', width=1.25)
##########################################################################


### Power spectrum #######################################################
if (args.powerspectrum=='y'):
    print "\nStarting power spectrum analysis..."
    bins_psd = int(args.psbinsize)
    # performing the DFT
    n_bins = len(lc_bin_t)
    k = np.arange(n_bins-1)
    frq = k/max(lc_bin_t)
    DFT = np.fft.fft(lc_bin_r) #/n
    t_bins = lc_bin_t[:-1]
    dt = t_bins[1] - t_bins[0]
    # grabbing only the relevant parts of frq and DFT
    half_n_bins = int((n_bins-1.0)/2.0)
    frq = frq[range(half_n_bins)]
    DFT = DFT[range(half_n_bins)]
    df = frq[1] - frq[0]
    # computing the PSD and background level
    PSD = (2.0*dt*abs(DFT)**2.0)/(n_bins*lc_bin_r_avg**2.0)
    PN_lev = 2.0*(lc_bin_r_avg + bg_bin_r_avg)/(lc_bin_r_avg**2.0)
    PSD -= PN_lev
    frq = frq[1:]+df/2. ; PSD = PSD[1:]

    # frq_bin, PSD_bin, PSD_bin_err = psdbin(bins_psd, frq, PSD)
    frq_bin, PSD_bin, PSD_bin_err = psdbin2(1.3, frq, PSD)

    print frq_bin
    print len(frq), len(frq_bin)
    print len(PSD), len(PSD_bin)

    ### Plot the power density spectrum
    plt.figure(2)
    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(y=PN_lev, color='k', dashes=[5,3], linewidth=1)
    ax.plot(frq, PSD, 'o', color='k', markersize=2, alpha=1.0)
    ax.errorbar(frq_bin, PSD_bin, yerr=PSD_bin_err, color='r', ecolor='r', fmt='o', markersize=5, markerfacecolor='none', linewidth=1)
    ax.fill_between(frq_bin, y1=PSD_bin-PSD_bin_err, y2=PSD_bin+PSD_bin_err, facecolor='r', alpha=0.25)
    ax.set_xlabel(r'Frequency [Hz]')
    ax.set_ylabel(r'Power [Hz$^{-1}$]')
    ax.set_xlim(left=1./(n_bins*dt), right=1./(2.*dt))
    ax.set_ylim(bottom=1e-3, top=1e3)
    # ax.set_ylim(bottom=min(PSD)-min(PSD)*.25, top=max(PSD)*1.5+max(PSD)*.25)
    ax.tick_params(axis='both', which='both', direction='in', top='on', right='on', width=1.25)
    # ax.legend(loc=3, ncol=2, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)
    # plt.savefig('lccor_soft_lc.png', bbox_inches='tight', dpi=300)
    print "...done!"
##########################################################################
print "*************************\n"
plt.show()
