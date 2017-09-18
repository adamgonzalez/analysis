#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:43:39 2017

@author: agonzalez
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally
plt.rc('font',family='serif')
plt.rc('text',usetex=True)


####################################################################################################
def error_avg(cts, err_cts, avg):
    ssq, s = 0., 0.
    n = len(cts)
    for i in range (0,n):
        ssq += (cts[i]-avg)**2.0
    ssq = ssq / (n-1.)
    s = np.sqrt(ssq/n)
    return s
####################################################################################################

####################################################################################################
def wmean(cts, err_cts):
    n = len(cts)
    w = 1./np.array(err_cts)
    top, bot, mean, err_mean = 0.,0.,0.,0.
    for i in range (0,n):
        top += cts[i]*w[i]**2.0
        bot += w[i]**2.0
    mean = top/bot
    err_mean = np.sqrt(1./bot)
    return mean, err_mean
####################################################################################################

####################################################################################################
def chisqcalc(dat, dat_err, avg):
    chisq, red_chisq = 0., 0.
    n = len(dat)
    for i in range (0,n):
        chisq += (dat[i]-avg)**2.0 / dat_err[i]**2.0
    red_chisq = chisq / (n-1.)
    return chisq, red_chisq
####################################################################################################

####################################################################################################
def fvarcalc(dat, dat_err, avg):
    Ssq, sqerr, Fvar, Fvar_err = 0., 0., 0., 0.
    n = len(dat)
    for i in range (0,n):
        Ssq += (dat[i]-avg)**2.0
        sqerr += dat_err[i]**2.0
    Ssq = Ssq / (n-1.)
    sqerr = sqerr / n
    Fvar = np.sqrt((Ssq-sqerr)/avg**2.0)
    Fvar_err = Ssq / (Fvar*np.sqrt(2.*n)*avg**2.0)
    return Fvar, Fvar_err
####################################################################################################

####################################################################################################
def mjd_calc(yr,mo,day,hr,mi,sec):
    a = np.floor((14.0 - mo) / 12.0)
    y = yr + 4800.0 - a
    m = mo + 12.0*a - 3.0
    JDN = day + np.floor((153.0*m+2.0)/5.0) + 365.0*y + np.floor(y/4.0) - np.floor(y/100.0) + np.floor(y/400.0) - 32045.0
    JN  = JDN + (hr-12.0)/24.0 + mi/1440.0 + sec/86400.0
    MJD = JN - 2400000.5
    return MJD
####################################################################################################

os.chdir("/Users/agonzalez/Documents/Research/Data/Mrk1501")

xmm = np.genfromtxt('0127110201pn_lccor_300-10000_1000s.qdp', skip_header=3) # either one data point at 0.2531 OR multiply each by 0.06815 , MJD start 51728.0 end 51729.0
#suz = np.genfromtxt('.n066l_xis0_comb_03-10_5760s.qdp', skip_header=3)       # either one data point at 0.3077 OR multiply each by 0.5094 , MJD start 55728.0
suz = np.genfromtxt('xis0xis3_03-10_5760s.qdp', skip_header=3)      # new data set
#suz[:,2], suz[:,3] = suz[:,2]/2., suz[:,3]/2.
swift = np.genfromtxt('XRT.lc', skip_header=2)


# print out all of the average count rates for each telescope and the error on the mean
#xmmavg, suzavg, swiftavg = np.average(xmm[:,2]), np.average(suz[:,2]), np.average(swift[:,3])
#e_xmmavg, e_suzavg, e_swiftavg = error_avg(xmm[:,2],xmm[:,3], xmmavg), error_avg(suz[:,2],suz[:,3], suzavg), error_avg(swift[:,3],swift[:,4], swiftavg)
xmmavg, e_xmmavg = wmean(xmm[:,2], xmm[:,3])
suzavg, e_suzavg = wmean(suz[:,2], suz[:,3])
swiftavg, e_swiftavg = wmean(swift[:,3], swift[:,4])
print 'XMM avg    = ', xmmavg, ' +/- ', e_xmmavg
print 'Suzaku avg = ', suzavg, ' +/- ', e_suzavg
print 'Swift avg  = ', swiftavg, ' +/- ', e_swiftavg
print ''

# compute the reduced chisq statistic for the average count rate fit to each light curve
chisq_xmm, red_chisq_xmm = chisqcalc(xmm[:,2], xmm[:,3], xmmavg)
chisq_suz, red_chisq_suz = chisqcalc(suz[:,2], suz[:,3], suzavg)
chisq_swift, red_chisq_swift = chisqcalc(swift[:,3], swift[:,4], swiftavg)
print 'xmm fit to avg: reduced chisq   = ', red_chisq_xmm
print 'suz fit to avg: reduced chisq   = ', red_chisq_suz
print 'swift fit to avg: reduced chisq = ', red_chisq_swift
print ''

# compute the fractional variability of each light curve given the average count rate computed above
fvar_xmm, e_fvar_xmm = fvarcalc(xmm[:,2], xmm[:,3], xmmavg)
fvar_suz, e_fvar_suz = fvarcalc(suz[:,2], suz[:,3], suzavg)
fvar_swift, e_fvar_swift = fvarcalc(swift[:,3], swift[:,4], swiftavg)
print 'fvar xmm   = ', fvar_xmm*100., ' +/- ', e_fvar_xmm*100., ' %'
print 'fvar suz   = ', fvar_suz*100., ' +/- ', e_fvar_suz*100., ' %'
print 'fvar swift = ', fvar_swift*100., ' +/- ', e_fvar_swift*100., ' %'
print ''

# bin up the swift light curve into MJD days (single observations)
lower = [0,  8, 12, 15, 16, 26, 29, 31, 32, 33, 35, 36, 37]
upper = [8, 12, 15, 15, 26, 29, 31, 31, 32, 35, 35, 36, 37]
weights=[8,  4,  3,  1, 10,  3,  2,  1,  1,  2,  1,  1,  1]
w, h = 5, 13
swiftbin = [[0 for x in range(w)] for y in range(h)]
swiftbin = np.array(swiftbin,dtype=float)

for i in range (0,h):
    if (i==3) or (i==7) or (i==8) or (i==10) or (i==11) or (i==12):
        swiftbin[i,0] = swift[lower[i],0]
        swiftbin[i,1] = swift[lower[i],1]
        swiftbin[i,2] = swift[lower[i],3]
        swiftbin[i,3] = swift[lower[i],4]
    else:
        swiftbin[i,0] = np.average(swift[lower[i]:upper[i],0])
        swiftbin[i,1] = np.average(swift[lower[i]:upper[i],1])
        swiftbin[i,2] = np.average(swift[lower[i]:upper[i],3])
        swiftbin[i,3] = error_avg(swift[lower[i]:upper[i],3],swift[lower[i]:upper[i],4], np.average(swift[lower[i]:upper[i],3]))

# compute the fractional variability of the binned swift light curve
swiftbinavg, e_swiftbinavg = wmean(swiftbin[:,2], swiftbin[:,3])
fvar_swiftbin, e_fvar_swiftbin = fvarcalc(swiftbin[:,2], swiftbin[:,3], swiftbinavg)
chisq_swiftbin, red_chisq_swiftbin = chisqcalc(swiftbin[:,2], swiftbin[:,3], swiftbinavg)
print 'Swiftbin avg                       = ', swiftbinavg, ' +/- ', e_swiftbinavg
print 'swiftbin fit to avg: reduced chisq = ', red_chisq_swiftbin
print 'fvar swiftbin                      = ', fvar_swiftbin*100., ' +/- ', e_fvar_swiftbin*100., ' %'
print ''



#### Plotting up the XMM and Suzaku short term light curves
#fig = plt.figure(figsize=(6,5))
#ax10 = fig.add_subplot(111)
#ax10.spines['top'].set_color('none')
#ax10.spines['bottom'].set_color('none')
#ax10.spines['left'].set_color('none')
#ax10.spines['right'].set_color('none')
#ax10.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
#
#xmm[:,0] = xmm[:,0] - xmm[0,0]
#ax11 = fig.add_subplot(211)
#ax11.errorbar(x=xmm[:,0], y=xmm[:,2], xerr=xmm[:,1], yerr=xmm[:,3], fmt='d', markersize=5, ecolor='g', capthick=1, linewidth=1, color='g', label='XMM', markerfacecolor='none')
#ax11.set_xlim(-xmm[0,1],xmm[14,0]+xmm[14,1])
#ax11.set_ylim(0.0,5.0) # lower bound 0.20
#ax11.tick_params(axis='both', which='both', direction='in', top='on', right='on')
#plt.legend(loc=3, ncol=2, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)
#ax11.get_legend()
#
#suz[:,0] = suz[:,0] - suz[0,0]
#ax12 = fig.add_subplot(212)
#ax12.axvline(suz[17,0]-suz[17,1], color='k', dashes=[5,3], linewidth=1)
#ax12.axvline(suz[22,0]-suz[22,1], color='k', dashes=[5,3], linewidth=1)
#ax12.errorbar(x=suz[:,0], y=suz[:,2], xerr=suz[:,1], yerr=suz[:,3], fmt='o', markersize=5, ecolor='r', capthick=1, linewidth=1, color='r', label='Suzaku', markerfacecolor='none')
#ax12.set_xlim(-suz[0,1],suz[29,0]+suz[29,1])
#ax12.set_ylim(0.0,0.75*2) # lower bound 0.20
#ax12.tick_params(axis='both', which='both', direction='in', top='on', right='on')
#ax12.text(.30,.35,'low\nflux',horizontalalignment='center',transform=ax12.transAxes)
#ax12.text(.87,.35,'high\nflux',horizontalalignment='center',transform=ax12.transAxes)
#plt.legend(loc=3, ncol=2, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)
#ax12.get_legend()
#
#ax10.set_xlabel('Time (s)')
#ax10.set_ylabel('Counts sec$^{-1}$ ($0.3 - 10$ keV)', labelpad=10)
#
##plt.savefig('/Users/agonzalez/Dropbox/Graduate/PhD/IIIZw2Paper2017/xmm_suz_lightcurves_zoomout.png', bbox_inches='tight', dpi=300)




### Combining all 3 lightcurve to form one data set
xmmmjd = xmm
xmmmjd[:,0] = xmm[:,0] - xmm[0,0]
xmmmjd[:,0] = xmmmjd[:,0]*1.16e-5
xmmmjd[:,1] = xmm[:,1]*1.16e-5
MJDxmm = mjd_calc(2000.0, 7.0, 3.0, 22.0, 44.0, 15.0)
xmmmjd[:,0] = xmmmjd[:,0] + MJDxmm
xmmmjd[:,2] = xmm[:,2]*0.06815
xmmmjd[:,3] = xmm[:,3]*0.06815
xmmmjdavg, e_xmmmjdavg = wmean(xmmmjd[:,2], xmmmjd[:,3])
xmmmjdT, e_xmmmjdT = wmean(xmmmjd[:,0], xmmmjd[:,1])
print 'XMMswift avg    = ', xmmmjdavg, ' +/- ', e_xmmmjdavg


suzmjd = suz
suzmjd[:,0] = suz[:,0] - suz[0,0]
suzmjd[:,0] = suzmjd[:,0]*1.16e-5
suzmjd[:,1] = suz[:,1]*1.16e-5
MJDsuz = mjd_calc(2011.0, 6.0, 14.0, 23.0, 42.0, 25.0)
suzmjd[:,0] = suzmjd[:,0] + MJDsuz
suzmjd[:,2] = suz[:,2]*0.5094
suzmjd[:,3] = suz[:,3]*0.5094
suzmjdavg, e_suzmjdavg = wmean(suzmjd[:,2], suzmjd[:,3])
suzmjdT, e_suzmjdT = wmean(suzmjd[:,0], suzmjd[:,1])
print 'Suzakuswift avg = ', suzmjdavg, ' +/- ', e_suzmjdavg


total_lc = [[swiftbin[0,0],swiftbin[0,1],swiftbin[0,2],swiftbin[0,3],swiftbin[0,4]]]
for i in range (1,len(swiftbin)):
    total_lc = np.concatenate((total_lc, [[swiftbin[i,0],swiftbin[i,1],swiftbin[i,2],swiftbin[i,3],swiftbin[i,4]]]))
total_lc = np.concatenate((total_lc, [[xmmmjdT, e_xmmmjdT, xmmmjdavg, e_xmmmjdavg, 0.]]))
total_lc = np.concatenate((total_lc, [[suzmjdT, e_suzmjdT, suzmjdavg, e_suzmjdavg, 0.]]))

# print out all of the average count rates for each telescope and the error on the mean
wm, e_wm = wmean(total_lc[:,2], total_lc[:,3])
print 'Wmean           = ', wm, ' +/- ', e_wm

# compute the reduced chisq statistic for the average count rate fit to each light curve
chisq_tot, red_chisq_tot = chisqcalc(total_lc[:,2], total_lc[:,3], wm)
print 'total fit to avg: reduced chisq = ', red_chisq_tot

# compute the fractional variability of each light curve given the average count rate computed above
fvar_tot, e_fvar_tot = fvarcalc(total_lc[:,2], total_lc[:,3], wm)
print 'fvar total = ', fvar_tot*100., ' +/- ', e_fvar_tot*100., ' %'



#plt.figure()
#plt.errorbar(x=xmmmjdT, y=xmmmjdavg, xerr=e_xmmmjdT, yerr=e_xmmmjdavg, fmt='d', markersize=7, ecolor='g', capthick=1, color='g', linewidth=1.0, label='XMM', markerfacecolor='none')
#plt.errorbar(x=suzmjdT, y=suzmjdavg/2.0, xerr=e_suzmjdT, yerr=e_suzmjdavg/2.0, fmt='o', markersize=7, ecolor='r', capthick=1, color='r', linewidth=1.0, label='Suzaku', markerfacecolor='none')
#plt.errorbar(x=swiftbin[:,0], y=swiftbin[:,2], xerr=swiftbin[:,1], yerr=swiftbin[:,3], fmt='s', markersize=7, ecolor='b', capthick=1, color='b', linewidth=1.0, label='Swift', markerfacecolor='none')
#plt.ylim(0.,1.)
#plt.xlabel('Time (MJD)')
#plt.ylabel('Counts sec$^{-1}$ (0.3 - 10 keV)')
#plt.legend(loc=2, ncol=1, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)
#
##plt.savefig('/Users/agonzalez/Dropbox/Graduate/Phd/IIIZw2Paper2017/all_lightcurves_swiftflux_noinset.png', bbox_inches='tight', dpi=300)
