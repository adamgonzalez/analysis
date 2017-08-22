#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: agonzalez
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.colors as colors
import os

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
from astropy.io import fits
from matplotlib.ticker import ScalarFormatter

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally
plt.rc('font',family='serif')
plt.rc('text',usetex=True)


os.chdir("/Users/agonzalez/Documents/Research/data/IZw1")


fits1, fits2 = fits.open('2768_fix5ksga.img'), fits.open('2769_fix5ksga.img')
data1, data2 = np.array(fits1[0].data), np.array(fits2[0].data)
# data1, data2 = fits1[0].data, fits2[0].data

fits3, fits4 = fits.open('2768_feLfix5ksga.img'), fits.open('2769_feLfix5ksga.img')
data3, data4 = np.array(fits3[0].data), np.array(fits4[0].data)
# data3, data4 = fits3[0].data, fits4[0].data

print 'Data1: Min= ', data1.min(), ' Max= ', data1.max(), ' Avg= ', np.average(data1)
print 'Data1: 0-point= ', -data1.min()/(-data1.min()+data1.max()), ' Mid-point= ', (-data1.min()+np.average(data1))/(-data1.min()+data1.max())
print 'Data2: Min= ', data2.min(), ' Max= ', data2.max(), ' Avg= ', np.average(data2)
print 'Data2: 0-point= ', -data2.min()/(-data2.min()+data2.max()), ' Mid-point= ', (-data2.min()+np.average(data2))/(-data2.min()+data2.max())
print 'Min Ratio 2/1= ', data2.min()/data1.min(), ' Max Ratio 2/1= ', data2.max()/data1.max()
print ''

print 'Data3: Min= ', data3.min(), ' Max= ', data3.max(), ' Avg= ', np.average(data3)
print 'Data3: 0-point= ', -data3.min()/(-data3.min()+data3.max()), ' Mid-point= ', (-data3.min()+np.average(data3))/(-data3.min()+data3.max())
print 'Data4: Min= ', data4.min(), ' Max= ', data4.max(), ' Avg= ', np.average(data4)
print 'Data3: 0-point= ', -data4.min()/(-data4.min()+data4.max()), ' Mid-point= ', (-data4.min()+np.average(data4))/(-data4.min()+data4.max())
print 'Min Ratio 3/4= ', data3.min()/data4.min(), ' Max Ratio 3/4= ', data3.max()/data4.max()
print ''


# making my own colourmap
cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.297448, 0.0, 0.0),
                   (0.393615, 0.75, 0.75),
                   (1.0, 1.5, 1.5)),
         'green': ((0.0, 0.0, 0.0),
                   (0.393615, 0.0, 0.0),
                   (1.0, 1.5, 1.5),
                   ),
         'blue':  ((0.0, 0.0, 0.0), #last 1.0
                   (0.393615, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }
# cdict2 = {'red':   ((0.0, 0.0, 0.0),
#                    (0.203303, 0.0, 0.0),
#                    (1.0, data2.max()/data1.max(), data2.max()/data1.max())),
#          'green': ((0.0, 0.0, 0.0),
#                    (0.425437, 0.0, 0.0),
#                    (1.0, 1.0, 1.0),
#                    ),
#          'blue':  ((0.0, 0.0, 0.0), #last data2.min()/data1.min()
#                    (0.425437, 0.0, 0.0),
#                    (1.0, 1.0, 1.0))
#         }
cdict2 = {'red':   ((0.0, 0.0, 0.0),
                   (0.203303, 0.0, 0.0),
                   (0.425437, 0.75*data2.max()/data1.max(), 0.75*data2.max()/data1.max()),
                   (1.0, 1.5*data2.max()/data1.max(), 1.5*data2.max()/data1.max())),
         'green': ((0.0, 0.0, 0.0),
                   (0.425437, 0.0, 0.0),
                   (1.0, 1.5*data2.max()/data1.max(), 1.5*data2.max()/data1.max()),
                   ),
         'blue':  ((0.0, 0.0, 0.0), #last 1.0
                   (0.425437, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }
cm1 = LinearSegmentedColormap('customcmap', cdict1)
cm2 = LinearSegmentedColormap('customcmap', cdict2)

cdict3 = {'red':   ((0.0, 0.0, 0.0),
                   (0.256278, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.256278, 0.0, 0.0),
                   (1.0, 1.5*data3.max()/data4.max(), 1.5*data3.max()/data4.max())),
         'blue':  ((0.0, 0.0, 0.0), #last data3.min()/data4.min()
                   (0.256278, 0.0, 0.0),
                   (0.315992, 0.75*data3.max()/data4.max(), 0.75*data3.max()/data4.max()),
                   (1.0, 1.5*data3.max()/data4.max(), 1.5*data3.max()/data4.max()))
        }
# cdict4 = {'red':   ((0.0, 0.0, 0.0),
#                    (0.253582, 0.0, 0.0),
#                    (1.0, 0.0, 0.0)),
#          'green': ((0.0, 0.0, 0.0),
#                    (0.253582, 0.0, 0.0),
#                    (1.0, 0.0, 0.0)),
#          'blue':  ((0.0, 0.0, 0.0), #last 1.0
#                    (0.253582, 0.0, 0.0),
#                    (1.0, 1.0, 1.0))
#         }
cdict4 = {'red':   ((0.0, 0.0, 0.0),
                   (0.253582, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.253582, 0.0, 0.0),
                   (1.0, 1.5, 1.5)),
         'blue':  ((0.0, 0.0, 0.0), #last 1.0
                   (0.253582, 0.0, 0.0),
                   (0.301326, 0.75, 0.75),
                   (1.0, 1.5, 1.5))
        }
cm3 = LinearSegmentedColormap('customcmap', cdict3)
cm4 = LinearSegmentedColormap('customcmap', cdict4)



plt.figure(1)

gs1 = gridspec.GridSpec(1,2, width_ratios=[0.98,1])
gs1.update(wspace=0.05, hspace=0)
ax1 = plt.subplot(gs1[0]) ; ax2 = plt.subplot(gs1[1])

# ax1 = plt.subplot(121)
img1 = ax1.imshow(data1, cmap=cm1, vmin=data1.min(), vmax=data1.max()) ; ax1.invert_yaxis()
# img1 = ax1.imshow(data1, cmap=cm1) ; ax1.invert_yaxis()
ax1.set_xticks([4,9,14,19,24]) ; ax1.set_xticklabels([5,10,15,20,25])
ax1.set_yticks([4,9,14,19,24]) ; ax1.set_yticklabels([5,6,7,8,9], verticalalignment='center', rotation='vertical')
ax1.set_xlabel('Time Segments') ; ax1.set_ylabel('Energy (keV)')
ax1.tick_params(axis='both', which='both', direction='in', top='on', right='on', color='white')
ax1.set_title(r'2768', size=18)
# plt.colorbar()


# ax2 = plt.subplot(122)
img2 = ax2.imshow(data2, cmap=cm2, vmin=data2.min(), vmax=data2.max()) ; ax2.invert_yaxis()
# img2 = ax2.imshow(data2, cmap=cm2) ; ax2.invert_yaxis()
ax2.set_xticks([4,9,14,19]) ; ax2.set_xticklabels([5,10,15,20])
ax2.set_yticks([4,9,14,19,24]) ; ax2.set_yticklabels([]) #ax2.set_yticklabels([5,6,7,8,9])
ax2.set_xlabel('Time Segments') #; ax2.set_ylabel('Energy (keV)')
ax2.tick_params(axis='both', which='both', direction='in', top='on', right='on', color='white')
ax2.set_title(r'2769', size=18)
cbar2 = plt.colorbar(img1, fraction=0.0473, pad=0.05, ticks=[-0.01, 0.0, 0.01, 0.02, 0.03])
# cbar2.ax.set_yticklabels([-1,0,1,2,3])
cbar2.ax.set_ylabel('photons cm$^{-2}$ 5000s$^{-1}$', rotation='270', labelpad=15.0)
# plt.colorbar()

# plt.savefig('FeK_residualmap_v2.ps', format='ps', bbox_inches='tight', dpi=300)


plt.figure(2)

gs2 = gridspec.GridSpec(1,3, width_ratios=[1,1,0.8])
gs2.update(wspace=0, hspace=0)
ax3 = plt.subplot(gs2[0]) ; ax4 = plt.subplot(gs2[1])

# ax3 = plt.subplot(121)
img3 = ax3.imshow(data3, cmap=cm3, vmin=data3.min(), vmax=data3.max()) ; ax3.invert_yaxis()
# img3 = ax3.imshow(data3, cmap=cm3) ; ax3.invert_yaxis()
ax3.set_xticks([4,9,14,19,24]) ; ax3.set_xticklabels([5,10,15,20,25])
ax3.set_yticks([9,19,29,39,49,59]) ; ax3.set_yticklabels([0.75,1.0,1.25,1.5,1.75,2.0], verticalalignment='center', rotation='vertical')
ax3.set_xlabel('Time Segments', size=14) ; ax3.set_ylabel('Energy (keV)')
ax3.tick_params(axis='both', which='both', direction='in', top='on', right='on', color='white')
ax3.set_title(r'2768', size=18)
# plt.colorbar()


# ax4 = plt.subplot(122)
img4 = ax4.imshow(data4, cmap=cm4, vmin=data4.min(), vmax=data4.max()) ; ax4.invert_yaxis()
# img4 = ax4.imshow(data4, cmap=cm4) ; ax4.invert_yaxis()
ax4.set_xticks([4,9,14,19]) ; ax4.set_xticklabels([5,10,15,20])
ax4.set_yticks([9,19,29,39,49,59]) ; ax4.set_yticklabels([]) #ax4.set_yticklabels([0.75,1.0,1.25,1.5,1.75,2.0])
ax4.set_xlabel('Time Segments', size=14) #; ax4.set_ylabel('Energy (keV)')
ax4.tick_params(axis='both', which='both', direction='in', top='on', right='on', color='white')
ax4.set_title(r'2769', size=18)
cbar4 = plt.colorbar(img3, fraction=0.105, pad=0.05, ticks=[-0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
# cbar4.ax.set_yticklabels([-2,0,2,4,6,8,10])
cbar4.ax.set_ylabel('photons cm$^{-2}$ 5000s$^{-1}$', rotation='270', labelpad=15.0)
# plt.colorbar()

# plt.savefig('FeL_residualmap_v2.ps', format='ps', bbox_inches='tight', dpi=300)


# plt.tight_layout()
plt.show()
