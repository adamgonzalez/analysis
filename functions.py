import numpy as np
import math

###################################################################################################
def wmean(cts, err_cts):
    n = len(cts)
    for j in range(0, n):
        if (cts[j]<=0.0) or (err_cts[j]<=0.0):
            cts[j] = 1e-3
            err_cts[j] = 1e-3/10.0
    w = 1./np.array(err_cts)
    top, bot, mean, err_mean = 0.,0.,0.,0.
    for i in range(0,n):
        top += cts[i]*w[i]**2.0
        bot += w[i]**2.0
    mean = top/bot
    err_mean = np.sqrt(1./bot)
    return mean, err_mean
###################################################################################################

###################################################################################################
def chisqcalc(dat, dat_err, avg):
    chisq, red_chisq = 0., 0.
    n = len(dat)
    for i in range(0,n):
        chisq += (dat[i]-avg)**2.0 / dat_err[i]**2.0
    red_chisq = chisq / (n-1.)
    return chisq, red_chisq
###################################################################################################

###################################################################################################
def fvarcalc(dat, dat_err, avg):
    Ssq, sqerr, Fvar, Fvar_err = 0., 0., 0., 0.
    n = len(dat)
    for i in range(0,n):
        Ssq += (dat[i]-avg)**2.0
        sqerr += dat_err[i]**2.0
    Ssq = Ssq / (n-1.)
    sqerr = sqerr / n
    Fvar = np.sqrt((Ssq-sqerr)/avg**2.0)
    Fvar_err = Ssq / (Fvar*np.sqrt(2.*n)*avg**2.0)
    return Fvar, Fvar_err
###################################################################################################

###################################################################################################
def lcbin(width, xvar, yvar, yerr):
    num_bins = int(np.ceil(max(xvar)/width))+1
    xvar_bin = np.arange(0, width*(num_bins), step=width)
    yvar_bin = np.zeros(num_bins-1)
    yerr_bin = np.zeros(num_bins-1)

    for i in range(0, num_bins-1):
        c = 0
        for j in range(0, len(xvar)):
            if (xvar_bin[i] <= xvar[j] < xvar_bin[i+1]):
                yvar_bin[i] += yvar[j]
                yerr_bin[i] += yerr[j]**2.0
                c += 1
        yvar_bin[i] /= c
        yerr_bin[i] = np.sqrt(yerr_bin[i])/c
    xerr_bin = width/2.

    # return xvar_bin[:-1]+xerr_bin, xerr_bin, yvar_bin, yerr_bin
    return xvar_bin[:-1], xerr_bin, yvar_bin, yerr_bin
###################################################################################################

###################################################################################################
def psdbin(n_bins, xvar, yvar):
    xvar_bin = np.logspace(np.log10(min(xvar)), np.log10(max(xvar)), n_bins)
    yvar_bin = np.zeros(n_bins)
    yerr_bin = np.zeros(n_bins)

    for i in range(0, n_bins-1):
        c = 0
        for j in range(0, len(xvar)):
            if (xvar_bin[i] <= xvar[j] < xvar_bin[i+1]):
                yvar_bin[i] += yvar[j]
                c += 1
        yvar_bin[i] /= c
        yerr_bin[i] = yvar_bin[i]/np.sqrt(c)#np.sqrt(yerr_bin[i])/c

    for k in range (0, n_bins-1):
        xvar_bin[k] = np.log10((np.power(10.0, xvar_bin[k])+np.power(10.0, xvar_bin[k+1])) / 2.0)

    return xvar_bin, yvar_bin, yerr_bin
###################################################################################################

###################################################################################################
def psdbin2(factor, xvar, yvar):
    n = len(xvar)
    bin_count = np.zeros(1)
    xvar_bin = np.zeros(1)
    yvar_bin = np.zeros(1)
    yerr_bin = np.zeros(1)

    xi = xvar[0]
    tracker = -1
    while (xi < max(xvar)):
        count = 0
        yvar_sum = 0.0
        for i in range(0, n):
            if (xi <= xvar[i] < factor*xi):
                yvar_sum += yvar[i]
                count += 1
                tracker += 1
        if (count < 2):
            yvar_sum = yvar[tracker]+yvar[tracker+1]
            tracker += 1
            count = 2
            bin_count = np.append(bin_count, [[count]])
            xvar_bin = np.append(xvar_bin, [[xi]])
            yvar_bin = np.append(yvar_bin, [[yvar_sum/count]])
            yerr_bin = np.append(yerr_bin, [[(yvar_sum/count)/np.sqrt(count)]])
            xi = xvar[tracker]
        elif (count >= 2):
            bin_count = np.append(bin_count, [[count]])
            xvar_bin = np.append(xvar_bin, [[xi]])
            yvar_bin = np.append(yvar_bin, [[yvar_sum/count]])
            yerr_bin = np.append(yerr_bin, [[(yvar_sum/count)/np.sqrt(count)]])
            xi = factor*xi

    xvar_bin = xvar_bin[1:]
    yerr_bin = yerr_bin/np.sqrt(len(xvar_bin))
    # for k in range(0, len(xvar_bin)-1):
    #     xvar_bin[k] = np.log10((np.power(10.0, xvar_bin[k])+np.power(10.0, xvar_bin[k+1])) / 2.0)

    return xvar_bin, yvar_bin[1:], yerr_bin[1:], bin_count[1:]
###################################################################################################

###################################################################################################
def cleanup(t, t_err, r, r_err):
    nan_t    = np.zeros(1)
    nan_t_err= np.zeros(1)
    nan_r    = np.zeros(1)
    nan_r_err= np.zeros(1)
    for k in range(0, len(t)):
        if (math.isnan(r[k])==True):
            if (math.isnan(r[k+1])==True):
                r[k] = (r[k-2] + r[k+2])/2.
            elif (math.isnan(r[k+2])==True):
                r[k] = (r[k-3] + r[k+3])/2.
            elif (math.isnan(r[k+3])==True):
                r[k] = (r[k-4] + r[k+4])/2.
            else:
                r[k] = (r[k-1] + r[k+1])/2.
            r_err[k] = np.average(r_err[:k-1])
            nan_t    = np.append(nan_t, [[t[k]]])
            nan_t_err= np.append(nan_t_err, [[t_err]])
            nan_r    = np.append(nan_r, [[r[k]]])
            nan_r_err= np.append(nan_r_err, [[r_err[k]]])

    return r, r_err, nan_t, nan_t_err, nan_r, nan_r_err
###################################################################################################
