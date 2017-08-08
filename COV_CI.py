# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 11:25:09 2017

@author: Adam
"""

import os
import numpy as np
import math
import cmath
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 
                   
# Function to compute the average count rate of a file (light curve or background)
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

os.chdir("/Users/agonzalez/Documents/Research/Data/Zwicky1")

#pic = '2769_COVvE_603.png'
#     ORBIT 2768
with open(name='2768/COV_CI/lc_covCI_2768_600.txt',mode='r') as lcfile:
#with open('2768/COV_CI/lc_covCI_2768_601.txt','r') as lcfile:
#with open('2768/COV_CI/lc_covCI_2768_602.txt','r') as lcfile:
#with open('2768/COV_CI/lc_covCI_2768_603.txt','r') as lcfile:
#with open('2768/COV_CI/lc_covCI_2768_604.txt','r') as lcfile:
    #     ORBIT 2769
#with open('2769/COV_CI/lc_covCI_2769_600.txt','r') as lcfile:
#with open('2769/COV_CI/lc_covCI_2769_601.txt','r') as lcfile:
#with open('2769/COV_CI/lc_covCI_2769_602.txt','r') as lcfile:
#with open('2769/COV_CI/lc_covCI_2769_603.txt','r') as lcfile:
#with open('2769/COV_CI/lc_covCI_2769_604.txt','r') as lcfile:
    lc_fnames = [line.rstrip('\n') for line in lcfile]
lcfile.close()
n_lc = len(lc_fnames)

#     ORBIT 2768
with open('2768/COV_CI/lcref_covCI_2768_600.txt','r') as reffile:
#with open('2768/COV_CI/lcref_covCI_2768_601.txt','r') as reffile:
#with open('2768/COV_CI/lcref_covCI_2768_602.txt','r') as reffile:
#with open('2768/COV_CI/lcref_covCI_2768_603.txt','r') as reffile:
#with open('2768/COV_CI/lcref_covCI_2768_604.txt','r') as reffile:
    #     ORBIT 2769
#with open('2769/COV_CI/lcref_covCI_2769_600.txt','r') as reffile:
#with open('2769/COV_CI/lcref_covCI_2769_601.txt','r') as reffile:
#with open('2769/COV_CI/lcref_covCI_2769_602.txt','r') as reffile:
#with open('2769/COV_CI/lcref_covCI_2769_603.txt','r') as reffile:
#with open('2769/COV_CI/lcref_covCI_2769_604.txt','r') as reffile:
    ref_fnames = [line.rstrip('\n') for line in reffile]
reffile.close()
n_ref = len(ref_fnames)

#     ORBIT 2768
with open('2768/COV_CI/bg_covCI_2768_600.txt','r') as bgfile:
#with open('2768/COV_CI/bg_covCI_2768_601.txt','r') as bgfile:
#with open('2768/COV_CI/bg_covCI_2768_602.txt','r') as bgfile:
#with open('2768/COV_CI/bg_covCI_2768_603.txt','r') as bgfile:
#with open('2768/COV_CI/bg_covCI_2768_604.txt','r') as bgfile:
    #     ORBIT 2769
#with open('2769/COV_CI/bg_covCI_2769_600.txt','r') as bgfile:
#with open('2769/COV_CI/bg_covCI_2769_601.txt','r') as bgfile:
#with open('2769/COV_CI/bg_covCI_2769_602.txt','r') as bgfile:
#with open('2769/COV_CI/bg_covCI_2769_603.txt','r') as bgfile:
#with open('2769/COV_CI/bg_covCI_2769_604.txt','r') as bgfile:
    bg_fnames = [line.rstrip('\n') for line in bgfile]
bgfile.close()
n_bg = len(bg_fnames)

#     ORBIT 2768
with open('2768/COV_CI/bgref_covCI_2768_600.txt','r') as refbgfile:
#with open('2768/COV_CI/bgref_covCI_2768_601.txt','r') as refbgfile:
#with open('2768/COV_CI/bgref_covCI_2768_602.txt','r') as refbgfile:
#with open('2768/COV_CI/bgref_covCI_2768_603.txt','r') as refbgfile:
#with open('2768/COV_CI/bgref_covCI_2768_604.txt','r') as refbgfile:
    #     ORBIT 2769
#with open('2769/COV_CI/bgref_covCI_2769_600.txt','r') as refbgfile:
#with open('2769/COV_CI/bgref_covCI_2769_601.txt','r') as refbgfile:
#with open('2769/COV_CI/bgref_covCI_2769_602.txt','r') as refbgfile:
#with open('2769/COV_CI/bgref_covCI_2769_603.txt','r') as refbgfile:
#with open('2769/COV_CI/bgref_covCI_2769_604.txt','r') as refbgfile:
    refbg_fnames = [line.rstrip('\n') for line in refbgfile]
refbgfile.close()
n_refbg = len(refbg_fnames)


#n_lc = 2
n_RUNS = n_lc


# set up all of the final output variables and the number of files to go thru
energy = [0.3, 0.45, 0.55, 0.7, 0.9, 1.25, 1.75, 3.0, 5.0, 7.0, 9.0]
energy = energy[:n_RUNS]

Df_LF = 4.0*pow(10,-4.0) - 1.0*pow(10,-4.0)
Df_MF = 1.5*pow(10,-3.0) - 0.4*pow(10,-3.0)
Df_HF = 4.0*pow(10,-3.0) - 2.0*pow(10,-3.0)

plt.rc('font',family='serif')

# do the stuff
for RUN in range (0,n_RUNS):
    print "RUN NUMBER: ", RUN+1

    lcfits = fits.open(lc_fnames[RUN])
    lcdata = lcfits[1].data
    lcfits.close()
    lc_t_raw = lcdata.field('TIME') ; lc_t_raw = lc_t_raw - lc_t_raw[0]
    lc_r_raw = lcdata.field('RATE')
    lc_e_raw = lcdata.field('ERROR')
    
    bgfits = fits.open(bg_fnames[RUN])
    bgdata = bgfits[1].data
    bgfits.close()
    bg_t_raw = bgdata.field('TIME') ; bg_t_raw = bg_t_raw - bg_t_raw[0]
    bg_r_raw = bgdata.field('RATE')
    bg_e_raw = bgdata.field('ERROR')
    
    reffits = fits.open(ref_fnames[RUN])
    refdata = reffits[1].data
    reffits.close()
    ref_t_raw = refdata.field('TIME') ; ref_t_raw = ref_t_raw - ref_t_raw[0]
    ref_r_raw = refdata.field('RATE')
    ref_e_raw = refdata.field('ERROR')
    
    refbgfits = fits.open(refbg_fnames[RUN])
    refbgdata = refbgfits[1].data
    refbgfits.close()
    refbg_t_raw = refbgdata.field('TIME') ; refbg_t_raw = refbg_t_raw - refbg_t_raw[0]
    refbg_r_raw = refbgdata.field('RATE')
    refbg_e_raw = refbgdata.field('ERROR')

    #print "Mean Energy = ", energy[RUN]
    lc_t, lc_r, lc_e, idx, avg_rate = data_cleaner(lc_t_raw, lc_r_raw, lc_e_raw) ; print "Average count rate = ", avg_rate
    bg_t, bg_r, bg_e, bg_idx, avg_bg_rate = data_cleaner(bg_t_raw, bg_r_raw, bg_e_raw) ; print "Average background rate = ", avg_bg_rate
    ref_t, ref_r, ref_e, ref_idx, avg_ref_rate = data_cleaner(ref_t_raw, ref_r_raw, ref_e_raw) ; print "Average ref count rate = ", avg_ref_rate
    refbg_t, refbg_r, refbg_e, refbg_idx, avg_refbg_rate = data_cleaner(refbg_t_raw, refbg_r_raw, refbg_e_raw) ; print "Average ref count rate = ", avg_refbg_rate

    # performing the DFT
    n_bins = len(lc_t)
    k = np.arange(n_bins-1)
    frq = k/max(lc_t)
    
    DFT = np.fft.fft(lc_r) #/n
    DFT_ref = np.fft.fft(ref_r)
    
    t_bins = lc_t[:-1]
    dt = t_bins[1] - t_bins[0]
    
    # grabbing only the relevant parts of frq and DFT
    half_n_bins = int((n_bins-1.0)/2.0)
    frq = frq[range(half_n_bins)]
    DFT = DFT[range(half_n_bins)]
    DFT_ref = DFT_ref[range(half_n_bins)]
    df = frq[1] - frq[0]
    
    # computing the PSD and background level
    PSD = (2.0*dt*abs(DFT)**2.0)/(n_bins*avg_rate**2.0)
    PN_lev = 2.0*(avg_rate + avg_bg_rate)/(avg_rate**2.0)
    
    PSD_ref = (2.0*dt*abs(DFT_ref)**2.0)/(n_bins*avg_ref_rate**2.0)
    PN_ref = 2.0*(avg_ref_rate + avg_refbg_rate)/(avg_ref_rate**2.0)
    
    if (RUN == 0):
        w, h = n_lc, half_n_bins
        r = [[0 for x in range(w)] for y in range(h)]
        phi = [[0 for x in range(w)] for y in range(h)]
        r_ref = [[0 for x in range(w)] for y in range(h)]
        phi_ref = [[0 for x in range(w)] for y in range(h)]
        CS = [[0 for x in range(w)] for y in range(h)]
        
    # working with the DFT values
    for i in range (0,half_n_bins):
        r[i][RUN], phi[i][RUN] = cmath.polar(DFT[i])
        r_ref[i][RUN], phi_ref[i][RUN] = cmath.polar(DFT_ref[i])

    # compute the cross spectrum
    for row in range (0,half_n_bins):
        CS[row][RUN] = (r[row][RUN]*r_ref[row][RUN]) * np.exp((-1.0*phi[row][RUN] + phi_ref[row][RUN])*1j)

    # bin up the PSD and CS
    C_LF = 0 ; C_MF = 0 ; C_HF = 0
    PSD_LF_avg = 0 ; PSD_MF_avg = 0 ; PSD_HF_avg = 0
    CS_LF_avg = 0 ; CS_MF_avg = 0 ; CS_HF_avg = 0

    for i in range (0,len(frq)):
        if (0.1e-3 <= frq[i] <= 0.4e-3):
            C_LF += 1
            PSD_LF_avg += PSD[i]
            CS_LF_avg += CS[i][RUN]
        if (0.4e-3 <= frq[i] <= 1.5e-3):
            C_MF += 1
            PSD_MF_avg += PSD[i]
            CS_MF_avg += CS[i][RUN]
        if (2e-3 <= frq[i] <= 4e-3):
            C_HF += 1
            PSD_HF_avg += PSD[i]
            CS_HF_avg += CS[i][RUN]

    PSD_LF_avg = PSD_LF_avg / C_LF
    PSD_MF_avg = PSD_MF_avg / C_MF
    PSD_HF_avg = PSD_HF_avg / C_HF
    
    CS_LF_avg = CS_LF_avg / C_LF
    CS_MF_avg = CS_MF_avg / C_MF
    CS_HF_avg = CS_HF_avg / C_HF

    C_ref_LF = 0 ; C_ref_MF = 0 ; C_ref_HF = 0
    PSD_ref_LF_avg = 0 ; PSD_ref_MF_avg = 0 ; PSD_ref_HF_avg = 0

    for i in range (0,len(frq)):
        if (0.1e-3 <= frq[i] <= 0.4e-3):
            C_ref_LF += 1
            PSD_ref_LF_avg += PSD_ref[i]
        if (0.4e-3 <= frq[i] <= 1.5e-3):
            C_ref_MF += 1
            PSD_ref_MF_avg += PSD_ref[i]
        if (2e-3 <= frq[i] <= 4e-3):
            C_ref_HF += 1
            PSD_ref_HF_avg += PSD_ref[i]
    
    PSD_ref_LF_avg = PSD_ref_LF_avg / C_ref_LF
    PSD_ref_MF_avg = PSD_ref_MF_avg / C_ref_MF
    PSD_ref_HF_avg = PSD_ref_HF_avg / C_ref_HF

    if (RUN ==0):
        COV_LF = np.zeros(n_lc)
        COV_MF = np.zeros(n_lc)
        COV_HF = np.zeros(n_lc)
            
    nsq_LF = ((PSD_LF_avg - PN_lev)*PN_ref + (PSD_ref_LF_avg - PN_ref)*PN_lev + PN_lev*PN_ref)/C_LF
    dfrq_LF = Df_LF
    COV_LF[RUN] = avg_rate * np.sqrt( dfrq_LF*(abs(CS_LF_avg)**2.0 - nsq_LF) / (PSD_ref_LF_avg - PN_ref) )
    
    nsq_MF = ((PSD_MF_avg - PN_lev)*PN_ref + (PSD_ref_MF_avg - PN_ref)*PN_lev + PN_lev*PN_ref)/C_MF
    dfrq_MF = Df_MF
    COV_MF[RUN] = avg_rate * np.sqrt( dfrq_MF*(abs(CS_MF_avg)**2.0 - nsq_MF) / (PSD_ref_MF_avg - PN_ref) )
    
    nsq_HF = ((PSD_HF_avg - PN_lev)*PN_ref + (PSD_ref_HF_avg - PN_ref)*PN_lev + PN_lev*PN_ref)/C_HF
    dfrq_HF = Df_HF
    COV_HF[RUN] = avg_rate * np.sqrt( dfrq_HF*(abs(CS_HF_avg)**2.0 - nsq_HF) / (PSD_ref_HF_avg - PN_ref) )
    

w, h = 4, len(energy)
M = [[0 for x in range(w)] for y in range(h)]
for i in range (0,len(energy)):
    M[i][0], M[i][1], M[i][2], M[i][3] = energy[i], COV_LF[i], COV_MF[i], COV_HF[i]

##outfile = open('E_COV_LMH_2768.txt','a')
#outfile = open('E_COV_LMH_2769.txt','a')
#np.savetxt(outfile,M)
#outfile.close()


plt.figure(1)
plt.loglog(energy, COV_LF, '-or', label = "LF")
plt.loglog(energy, COV_MF, '-og', label = "MF")
plt.loglog(energy, COV_HF, '-ob', label = "HF")
plt.xlabel("Energy [keV]")
plt.ylabel('keV$^2$ (Photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$)')
plt.xlim(0.20,10.0)
plt.legend(loc=3,labelspacing=0.1,fontsize=16)
plt.show()
#plt.savefig(pic,bbox_inches='tight')