#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:42:58 2017

@author: agonzalez
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
import matplotlib
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally
plt.rc('font',family='serif')
plt.rc('text',usetex=True)


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

####################################################################################################
def error_avg(cts,err_cts):
    n = len(cts)
    sr = np.zeros(n)
    for i in range (0,n):
        for j in range (0,n):
            if (i!=j):
                sr[i] += cts[j]
        sr[i] = (sr[i] / n)**2.0 * err_cts[i]**2.0
    st = np.sqrt(sum(sr))
    return st
####################################################################################################

####################################################################################################
# Plot x and y with labels (trying to do it)
def plottertron(x, x_error, y, y_error, line_color, line_label, xscale, yscale, legyn, legloc, style, shp):
    ax.set_xlabel('Time (MJD)')
    ax.set_ylabel('Counts / sec ($0.3 - 10$ keV)')
    if (xscale == 'log'):
        ax.set_xscale('log')
    if (yscale == 'log'):
#        ax.set_ylabel('Normalized Counts (s$^{-1}$ keV$^{-1}$)')
#        ax.set_ylabel('keV$^{2}$ (Photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$)')   # use this when plotting eeuf 
        ax.set_yscale('log')
    if (style == 'line'):
        ax.errorbar(x, y, xerr=x_error, yerr=y_error, ecolor=line_color, capthick=1, color=line_color, label=line_label)
    elif (style == 'marker'):
        ax.errorbar(x, y, xerr=x_error, yerr=y_error, fmt=shp, markersize=5, ecolor=line_color, capthick=1, linewidth = 1, color=line_color, label=line_label, markerfacecolor='none')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.show()
    if (legyn == 1):
#        plt.legend(loc=legloc, ncol=2, labelspacing=0.1, fontsize=16)
        plt.legend(loc=legloc, ncol=1, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)
        ax.get_legend()
####################################################################################################



### Read in the data from the .qdp files made by wdata in iplot on XSPEC
xmm = np.genfromtxt('0127110201pn_lccor_300-10000_1000s.qdp', skip_header=3) # either one data point at 0.2531 OR multiply each by 0.06815 , MJD start 51728.0 end 51729.0
suz = np.genfromtxt('n066l_xis0_comb_03-10_5760s.qdp', skip_header=3)       # either one data point at 0.3077 OR multiply each by 0.5094 , MJD start 55728.0
swift = np.genfromtxt('XRT.lc', skip_header=2)

##print xmm[0,0]
#xmmmjd = xmm
#xmmmjd[:,0] = xmm[:,0] - xmm[0,0]
#xmmmjd[:,0] = xmmmjd[:,0]*1.16e-5
#xmmmjd[:,1] = xmm[:,1]*1.16e-5
#MJDxmm = mjd_calc(2000.0, 7.0, 3.0, 22.0, 44.0, 15.0)
#xmmmjd[:,0] = xmmmjd[:,0] + MJDxmm
#xmmmjd[:,2] = xmm[:,2]*0.06815
#xmmmjd[:,3] = xmm[:,3]*0.06815
#
#suzmjd = suz
#suzmjd[:,0] = suz[:,0] - suz[0,0]
#suzmjd[:,0] = suzmjd[:,0]*1.16e-5
#suzmjd[:,1] = suz[:,1]*1.16e-5
#MJDsuz = mjd_calc(2011.0, 6.0, 14.0, 23.0, 42.0, 25.0)
#suzmjd[:,0] = suzmjd[:,0] + MJDsuz
#suzmjd[:,2] = suz[:,2]*0.5094
#suzmjd[:,3] = suz[:,3]*0.5094
#
#
### Doing some binning things
#xmmbin = np.zeros(5)
#xmmbin[0], xmmbin[1], xmmbin[2], xmmbin[3], xmmbin[4] = np.average(xmmmjd[:,0]), np.average(xmmmjd[:,1]), np.average(xmmmjd[:,2]), error_avg(xmmmjd[:,2],xmmmjd[:,3]), np.average(xmmmjd[:,4])
#    
#suzbin = np.zeros(5)
#suzbin[0], suzbin[1], suzbin[2], suzbin[3], suzbin[4] = np.average(suzmjd[:,0]), np.average(suzmjd[:,1]), np.average(suzmjd[:,2]), error_avg(suzmjd[:,2],suzmjd[:,3]), np.average(suzmjd[:,4])
#
#lower = [0,8,12,15,16,26,29,31,32,33,35,36]
#upper = [8,12,15,15,26,29,31,31,32,35,35,38] 
#w, h = 5, 12
#swiftbin = [[0 for x in range(w)] for y in range(h)]
#swiftbin = np.array(swiftbin,dtype=float)
#
#for i in range (0,h):
#    if (i==3) or (i==7) or (i==8) or (i==10):
#        swiftbin[i,0] = swift[lower[i],0]
#        swiftbin[i,1] = swift[lower[i],1]
#        swiftbin[i,2] = swift[lower[i],3]
#        swiftbin[i,3] = swift[lower[i],4]
#    else:
#        swiftbin[i,0] = np.average(swift[lower[i]:upper[i],0])
#        swiftbin[i,1] = np.average(swift[lower[i]:upper[i],1])
#        swiftbin[i,2] = np.average(swift[lower[i]:upper[i],3])
##        swiftbin[i,3] = error_avg(swift[lower[i]:upper[i],3],swift[lower[i]:upper[i],4])
#
#
#print 'XMM avg = ', np.average(xmm[:,2]), ' +/- ', error_avg(xmm[:,2],xmm[:,3])
#print 'Suzaku avg = ', np.average(suz[:,2]), ' +/- ', error_avg(suz[:,2],suz[:,3])
#print 'Swift avg = ', np.average(swiftbin[:,2]), ' +/- ', error_avg(swiftbin[:,2],swiftbin[:,3])
#
##### Plot up the data 
#fig = plt.figure()
#ax = fig.add_subplot(111)
#
##### XMM Data
##plottertron(xmmmjd[:,0], xmmmjd[:,1], xmmmjd[:,2], xmmmjd[:,3], 'g', 'XMM', 'linear', 'linear', 0, 2, 'marker', 'D')
#plottertron(xmmbin[0], xmmbin[1], xmmbin[2], xmmbin[3], 'g', 'XMM', 'linear', 'linear', 0, 2, 'marker', 'D')
##ax.fill_between(x=[xmm[0,0],swiftbin[-1,0]], y1=np.average(xmm[:,2])+error_avg(xmm[:,2],xmm[:,3]), y2=np.average(xmm[:,2])-error_avg(xmm[:,2],xmm[:,3]), alpha=0.5, color='g')
#ax.tick_params(axis='both', which='both', direction='in', top='on', right='on')
#
#
##### Suzaku
##plottertron(suzmjd[:,0], suzmjd[:,1], suzmjd[:,2], suzmjd[:,3], 'r', 'Suzaku', 'linear', 'linear', 0, 2, 'marker', 'o')
#plottertron(suzbin[0], suzbin[1], suzbin[2], suzbin[3], 'r', 'Suzaku', 'linear', 'linear', 0, 2, 'marker', 'o')
##ax.fill_between(x=[xmm[0,0],swiftbin[-1,0]], y1=np.average(suz[:,2])+error_avg(suz[:,2],suz[:,3]), y2=np.average(suz[:,2])-error_avg(suz[:,2],suz[:,3]), alpha=0.5, color='r')
#ax.tick_params(axis='both', which='both', direction='in', top='on', right='on')
#
#
##### Swift
##plottertron(swift[:,0], swift[:,1], swift[:,3], swift[:,4], 'b', 'Swift', 'linear', 'linear', 1, 2, 'marker', 's')
#plottertron(swiftbin[:3,0], swiftbin[:3,1], swiftbin[:3,2], swiftbin[:3,3], 'b', 'Swift', 'linear', 'linear', 1, 2, 'marker', 's')
#ax.errorbar(x=swiftbin[3,0], y=swiftbin[3,2], xerr=swiftbin[3,1], yerr=swiftbin[3,3], fmt='s', markersize=7, ecolor='b', capthick=1, linewidth=1, color='b', uplims=True)
#plottertron(swiftbin[4:,0], swiftbin[4:,1], swiftbin[4:,2], swiftbin[4:,3], 'b', 'Swift', 'linear', 'linear', 0, 2, 'marker', 's')
##ax.fill_between(x=[xmm[0,0],swiftbin[-1,0]], y1=np.average(swiftbin[:,2])+error_avg(swiftbin[:,2],swiftbin[:,3]), y2=np.average(swiftbin[:,2])-error_avg(swiftbin[:,2],swiftbin[:,3]), alpha=0.5, color='b')
#ax.tick_params(axis='both', which='both', direction='in', top='on', right='on')
#ax.set_ylim(0.0,0.9)
#
###plt.savefig('../../LaTeX/IIIZw2/all_lightcurves_swiftflux_noinset.png',bbox_inches='tight',dpi=300)
##plt.savefig('/Users/agonzalez/Dropbox/Graduate/PhD/IIIZw2Paper2017/all_lightcurves_swiftflux_noinset.png', bbox_inches='tight', dpi=300)



#### Setting up the insets
#ax.add_patch(patches.Rectangle((min(xmmmjd[:,0])-100, 0.220), max(xmmmjd[:,0])-min(xmmmjd[:,0])+200, 0.07, fill=False, edgecolor='0.50'))
#ax.add_patch(patches.Rectangle((min(suzmjd[:,0])-100, 0.2), max(suzmjd[:,0])-min(suzmjd[:,0])+210, 0.17, fill=False, edgecolor='0.50'))
#
##                   left bot  w   h
#ax2 = fig.add_axes([0.16,0.45,0.25,0.2])
#ax2.errorbar(x=xmmmjd[:,0], y=xmmmjd[:,2], xerr=xmmmjd[:,1], yerr=xmmmjd[:,3], fmt='^', markersize=3, ecolor='g', capthick=1, linewidth=1, color='g', label='XMM')
#ax2.spines['bottom'].set_color('k') ; ax2.spines['top'].set_color('k') ; ax2.spines['right'].set_color('k') ; ax2.spines['left'].set_color('k')
#ax.plot([min(xmmmjd[:,0])-90, 51630], [0.29, 0.44], color='0.50', linewidth=1)
#ax.plot([max(xmmmjd[:,0])+100, 53840], [0.29, 0.44], color='0.50', linewidth=1)
##ax2.xaxis.set_visible(False)
#ax2.yaxis.set_visible(False)
#ax2.tick_params(labelsize=10)
#ax2.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
#ax2.tick_params(axis='both', which='both', direction='in', top='on', right='on')
#
#
##                   left bot  w   h
#ax3 = fig.add_axes([0.62,0.5,0.25,0.2])
#ax3.errorbar(x=suzmjd[:,0], y=suzmjd[:,2], xerr=suzmjd[:,1], yerr=suzmjd[:,3], fmt='o', markersize=3, ecolor='r', capthick=1, linewidth=1, color='r', label='Suzaku')
#ax3.spines['bottom'].set_color('k') ; ax3.spines['top'].set_color('k') ; ax3.spines['right'].set_color('k') ; ax3.spines['left'].set_color('k')
#ax.plot([min(suzmjd[:,0])-90, 55770], [0.37, 0.51], color='0.50', linewidth=1)
#ax.plot([max(suzmjd[:,0])+100, 57965], [0.37, 0.505], color='0.50', linewidth=1)
##ax3.xaxis.set_visible(False)
#ax3.yaxis.set_visible(False)
#ax3.tick_params(labelsize=10)
#ax3.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))
#ax3.tick_params(axis='both', which='both', direction='in', top='on', right='on')


#axins1 = zoomed_inset_axes(ax, 2.0, loc=9)
#axins1.errorbar(xmmmjd[:,0], xmmmjd[:,2], xmmmjd[:,1], xmmmjd[:,3], fmt='D', markersize=3, ecolor='g', capthick=1, linewidth=1, color='g', label='XMM')
#axins1.set_xlim(min(xmmmjd[:,0]),max(xmmmjd[:,0]))
#axins1.set_ylim(0.2,0.3)
#axins1.xaxis.set_visible(False)
#axins1.yaxis.set_visible(False)
#mark_inset(ax,axins1, loc1=2, loc2=4, fc="none", ec="0.5")

#plt.savefig('../../LaTeX/IIIZw2/all_lightcurves_swiftflux.png',bbox_inches='tight',dpi=300)




#### Small subfigures of the lightcurves ####################################################################
fig = plt.figure(figsize=(6,5))
ax10 = fig.add_subplot(111)
ax10.spines['top'].set_color('none')
ax10.spines['bottom'].set_color('none')
ax10.spines['left'].set_color('none')
ax10.spines['right'].set_color('none')
ax10.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

xmm[:,0] = xmm[:,0] - xmm[0,0]
#xmm[:,2] = xmm[:,2]*0.06815
#xmm[:,3] = xmm[:,3]*0.06815
#fig = plt.figure(figsize=(6,2.5))
ax11 = fig.add_subplot(211)
ax11.errorbar(x=xmm[:,0], y=xmm[:,2], xerr=xmm[:,1], yerr=xmm[:,3], fmt='D', markersize=0, ecolor='g', capthick=1, linewidth=1, color='g', label='XMM', markerfacecolor='none')
ax11.set_xlim(-xmm[0,1],xmm[14,0]+xmm[14,1])
ax11.set_ylim(0.0,5.0) # lower bound 0.20
ax11.tick_params(axis='both', which='both', direction='in', top='on', right='on')
#ax11.set_xlabel('Time (s)')
#ax11.set_ylabel('Counts / sec ($0.3 - 10$ keV)')
plt.legend(loc=3, ncol=2, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)
ax11.get_legend()

#print 95000 + suz[0,0]
suz[:,0] = suz[:,0] - suz[0,0]
#suz[:,2] = suz[:,2]*0.5094
#suz[:,3] = suz[:,3]*0.5094
#fig = plt.figure(figsize=(6,2.5))
ax12 = fig.add_subplot(212)
#ax12.axvline(x=95000)
#ax12.axvline(suz[16,0]+suz[16,1],color='g')
ax12.axvline(suz[17,0]-suz[17,1],color='k',linestyle='-',linewidth=1)
#ax12.axvline(suz[21,0]+suz[21,1],color='g')
ax12.axvline(suz[22,0]-suz[22,1],color='k',linestyle='-',linewidth=1)
ax12.errorbar(x=suz[:,0], y=suz[:,2], xerr=suz[:,1], yerr=suz[:,3], fmt='o', markersize=0, ecolor='r', capthick=1, linewidth=1, color='r', label='Suzaku', markerfacecolor='none')
ax12.set_xlim(-suz[0,1],suz[29,0]+suz[29,1])
ax12.set_ylim(0.0,0.75) # lower bound 0.20
ax12.tick_params(axis='both', which='both', direction='in', top='on', right='on')
#ax12.set_xlabel('Time (s)')
#ax12.set_ylabel('Counts / sec ($0.3 - 10$ keV)')
ax12.text(.30,.4,'low',horizontalalignment='center',transform=ax12.transAxes)
ax12.text(.87,.4,'high',horizontalalignment='center',transform=ax12.transAxes)
plt.legend(loc=3, ncol=2, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)
ax12.get_legend()

ax10.set_xlabel('Time (s)')
ax10.set_ylabel('Counts / sec ($0.3 - 10$ keV)', labelpad=10)

#plt.savefig('../../LaTeX/IIIZw2/xmm_suz_lightcurves_zoomout.png',bbox_inches='tight',dpi=300)
#plt.savefig('/Users/agonzalez/Dropbox/Graduate/PhD/IIIZw2Paper2017/xmm_suz_lightcurves_zoomout.png', bbox_inches='tight', dpi=300)
