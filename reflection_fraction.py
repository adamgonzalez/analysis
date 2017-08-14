# -*- coding: utf-8 -*-
"""
@author: adamg
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally
plt.rc('font',family='serif')
plt.rc('text',usetex=True)

# this script is designed to fit the reflection fraction curves for different beaming values of point sources at different heights

plt.figure(1)

for j in range (0,6):

    b = [0.0, 0.25, 0.50, 0.75, 0.90, 1.0]

    if (j==0):
        h = 1.5
        b = [0.0, 0.125, 0.25, 0.375, 0.50, 0.75, 0.90, 1.0]
        disc = [3275158., 3358125., 3440022., 3530441., 3589077., 3468784., 2683029., 0.]
        esc = [280490., 311332., 364215., 442736., 564990., 1113312., 2186816., 1.]
        name = "$z$=1.5$r_g$" ; c = 'red' ; style = [5,3]

    elif (j==1):
        h = 2.0
        disc = [3760825., 3711535., 3448920., 2686783., 1565926., 0.]
        esc = [634862., 867866., 1309359., 2238717., 3454230., 1.]
        name = "$z$=2$r_g$" ; c = "green" ; style = [10,4,2,4]
    elif (j==2):
        h = 3.0
        disc = [3703528., 3357163., 2771914., 1790728., 862943., 0.]
        esc = [1117200., 1551011., 2204971., 3242121., 4199993., 1.]
        name = "$z$=3$r_g$" ; c = "blue" ; style = [2,2]
    elif (j==3):
        h = 5.0
        disc = [3384878., 2840337., 2136191., 1220103., 531397., 0.]
        esc = [1601226., 2174811., 2901166., 3837081., 4534824., 1.]
        name = "$z$=5$r_g$" ; c = 'orange' ; style = [10,2]
    elif (j==4):
        h = 10.0
        disc = [3011576., 2384863., 1679540., 890378., 370222., 0.]
        esc = [2024686., 2660208., 3370244., 4171421., 4700730., 1.]
        name = "$z$=10$r_g$" ; c = 'magenta' ; style = [1,1]
    elif (j==5):
        h = 20.0
        disc = [2769920., 2129048., 1456541., 744354., 302192., 0.]
        esc = [2257173., 2901075., 3582326., 4307499., 4763525., 1.]
        name = "$z$=20$r_g$" ; c = 'cyan' ; style = [10,2,3,2,3,2]

    refl_frac = np.zeros(len(b))

    for i in range (0,len(b)):
         refl_frac[i] = disc[i]/esc[i]

    # plt.plot(b, refl_frac/refl_frac[0], dashes = [1,0.1], color = c, label = name, linewidth = 2)
    plt.plot(b, refl_frac, color = c, label = name, linewidth = 2)

    a = 0.998
    z = h
    beta = np.linspace(0,1,100)
    ref_fit = np.zeros(len(beta))
    for i in range (0, len(beta)):
        mu_in = (2.0 - z**2.0)/(z**2.0 + a**2.0)
        mu_out = (2.0*z)/(z**2.0 + a**2.0)
        mu_in_p = (mu_in - beta[i])/(1.0 - beta[i]*mu_in)
        mu_out_p = (mu_out - beta[i])/(1.0 - beta[i]*mu_out)
        ref_fit[i] = (mu_out_p - mu_in_p)/(1.0 - mu_out_p)
    # plt.plot(beta, ref_fit/ref_fit[0], dashes = [5,3], color = c, linewidth = 2)
    plt.plot(beta, ref_fit, dashes = [3,3], color = c, linewidth = 2)


plt.axhline(y=0.54, color='k', linewidth=1.0)
plt.fill_between(x=b, y1=0.54-0.04, y2=0.54+0.04, color='k', alpha=0.25)
plt.xlim(b[0],b[-1])
plt.ylim(0.,1.0)

plt.rc('font',family='serif')
plt.xlabel(r"Velocity, $\beta$")
#plt.ylabel("Corrected Reflection Fraction, $R/R_{\mathrm{0}}$")
plt.ylabel("Reflection Fraction, $R$")
plt.legend(loc='best', ncol=1, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)


#plt.savefig("../Plots/R_vs_beta_reffit.png",bbox_inches='tight')
#plt.savefig("../Plots/RdR0_vs_beta_reffit_shaded.png",bbox_inches='tight')
plt.show()


plt.figure(2)
plt.rc('font',family='serif')

for j in range (0,5):
   height_tot = [1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0]

   if (j == 0):
       b = 0.0
       src_tot = [11.6765588791, 5.92384644222, 3.31500895095, 2.50504, 2.11392895194, 1.73574, 1.48742866795, 1.22716335877] ; c = 'red' ; name = r"$\beta$ = 0.0"
   elif (j == 1):
       b = 0.25
       src_tot = [9.44503109427, 4.27662219744, 2.16449980045, 1.57423, 1.30601555721, 1.05404, 0.896494935734, 0.73388244013] ; c = 'green' ; name = r"$\beta$ = 0.25"
   elif (j == 2):
       b = 0.50
       src_tot = [6.35246110551, 2.63405223472, 1.25712038843, 0.896848, 0.736321534169, 0.58955, 0.498343740097, 0.406590857448] ; c = 'blue' ; name = r"$\beta$ = 0.50"
   elif (j == 3):
       b = 0.75
       src_tot = [3.11573395418, 1.20014410039, 0.552332254102, 0.389345, 0.317976868354, 0.253084, 0.213447168243, 0.172804218875] ; c = 'orange' ; name = r"$\beta$ = 0.75"
   elif (j == 4):
       b = 0.90
       src_tot = [1.22691118046, 0.453335765134, 0.205462961486, 0.144132565, 0.117181394471, 0.0934901, 0.0787584056093, 0.0634387349704] ; c = 'magenta' ; name = r"$\beta$ = 0.90"

   height = height_tot[0:]
   src = src_tot[0:]

   plt.scatter(height, src, marker = 'o', color = c, s = 20.0)
   plt.plot(height, src, dashes = [1,0.1], color = c, label = name, linewidth=1)

   a = 0.998
   z = np.linspace(1.5,20,100)
   beta = [0.0, 0.25, 0.50, 0.75, 0.90]
   ref_fit = np.zeros(len(z))
   for i in range (0, len(z)):
       mu_in = (2.0 - z[i]**2.0)/(z[i]**2.0 + a**2.0)
       mu_out = (2.0*z[i])/(z[i]**2.0 + a**2.0)
       mu_in_p = (mu_in - beta[j])/(1.0 - beta[j]*mu_in)
       mu_out_p = (mu_out - beta[j])/(1.0 - beta[j]*mu_out)
       ref_fit[i] = (mu_out_p - mu_in_p)/(1.0 - mu_out_p)
   plt.plot(z,ref_fit, color = c, dashes = [5,3], linewidth=2)


plt.axhline(y=0.54, color='k', linewidth=1.0)
plt.fill_between(x=z, y1=0.54-0.04, y2=0.54+0.04, color='k', alpha=0.25)
plt.xlim(z[0],z[-1])
plt.ylim(0.,1.0)

plt.xlabel("Height, $z$")
plt.ylabel("Reflection Fraction, $R$")
plt.legend(loc=1, ncol=1, labelspacing=0.1, fontsize=16, handletextpad=0.1, fancybox=False, frameon=False)


# plt.savefig("../Plots/R_vs_z_allbeta_reffit_1.5rg.png",bbox_inches='tight')
plt.show()
