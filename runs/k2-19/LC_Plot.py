### Edit this range to plot a different segment of data
trange = [[2000., 2050.],[2050., 2100]]
#

import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
import sys

outname = sys.argv[1]

colorlist = ['b', 'r', 'g', 'y', 'c', 'm', 'midnightblue', 'yellow']
letters = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k']

def plotlc(time, flux, error, model, outname, zoomrange=None):

    plt.figure(figsize=(14,8))

    plt.errorbar(time, flux, yerr=error, c='k')

    plt.plot(time, model, c=colorlist[0], zorder=1000)


    top = np.max(flux + 0.0001)
    bot = np.min([np.min(flux), np.min(model)])
    timemin = np.min(time)
    timemax = np.max(time)

    tbv = glob.glob("./tbv[0-9][0-9]_[0-9][0-9].out")
    for i in range(len(tbv)):
        f = tbv[i]
        ttimes = pd.read_csv(f, sep='  ', header=None)
        for j in range(len(ttimes[1])):
            label = 'Planet {}'.format(letters[i])
            if j > 0:
                label = None
            plt.plot([ttimes[1][j], ttimes[1][j]], [top, top+1e-4],
                     c=colorlist[i], marker="None", label=label)

    plt.xlabel('Time (days)', fontsize=20)
    plt.ylabel('Flux', fontsize=20)
    savestr = 'lcplot'

    
    plt.xlim(timemin-1, timemax + 1)
    plt.ylim(bot-1e4/2, top+1e-4/2)
    plt.legend(fontsize=16)
    plt.savefig(outname+'_all.png',dpi=300)


    if zoomrange is not None:
        for z in zoomrange:
            plt.figure(figsize=(14,8))

            plt.errorbar(time, flux, yerr=error, c='k')

            plt.plot(time, model, c=colorlist[0], zorder=1000)

            bot = np.min([np.min(flux), np.min(model)])
            top = np.max(flux + 0.0001)

            tbv = glob.glob("./tbv[0-9][0-9]_[0-9][0-9].out")
            for i in range(len(tbv)):
                f = tbv[i]
                ttimes = pd.read_csv(f, sep='  ', header=None)
                for j in range(len(ttimes[1])):
                    label = 'Planet {}'.format(letters[i])
                    if j > 0:
                        label = None
                    plt.plot([ttimes[1][j], ttimes[1][j]], [top, top+1e-4],
                         c=colorlist[i], marker="None", label=label)

            plt.xlabel('Time (days)', fontsize=20)
            plt.ylabel('Flux', fontsize=20)
            plt.legend(fontsize=16)

            plt.ylim(bot-1e-4/2, top+1e-4/2)
            plt.xlim(z[0], z[1])
            plt.savefig(outname+'_zoom_{}.png'.format(z), dpi=300)
            
            plt.close()
####

lcdatafile = glob.glob("./lc_*.lcout")
if len(lcdatafile) == 0:
    print("This script must be run in the directory containing the lc_RUNNAME.lcout")
    print("    file produced with the 'lcout' command. No such file was not found here.")
    print("    Aborting")
    exit()
if len(lcdatafile) > 1:
    print("Warning: Multiple lc_RUNNAME.lcout files found in this directory")
    print("    The default behavior is to plot the first one alphabetically")

lcdata = np.loadtxt(lcdatafile[0])

time = lcdata[:,0]
flux = lcdata[:,1]
model = lcdata[:,2]
err = lcdata[:,3]

####
print('Doing the lc plot')
plotlc(time, flux, err, model, outname, trange)


def Plot_Phasefold(time, flux, err, outname):


    print('Phasefolding')
    tbvfilelist = glob.glob("./tbv[0-9][0-9]_[0-9][0-9].out")
    nfiles = len(tbvfilelist)
    npl = nfiles

    if npl == 0:
        print("Error: no tbvXX_YY.out files found in this directory")
        exit()

    f, axes = plt.subplots(npl, 1, figsize=(14,5*npl))
    try:
        axes = list(axes)
    except:
        axes = [axes]

    transtimes = [[] for i in range(npl)]
    nums = [[] for i in range(npl)]
    for i in range(nfiles):
        data = np.loadtxt(tbvfilelist[i])
        tt = data[:,1]
        transtimes[i] = tt
        nn = data[:,0]
        nums[i] = nn

    phasewidth = [0.4 for i in range(nfiles)]
    for i in range(nfiles):
        if len(transtimes[i]) > 1:
            meanper, const = np.linalg.lstsq(np.vstack([nums[i], np.ones(len(nums[i]))]).T, transtimes[i], rcond=None)[0]
            # 3x the duration of an edge-on planet around the sun
            phasewidth[i] = 3.*(13./24.) * ((meanper/365.25)**(1./3.))
            collisionwidth = [pwi for pwi in phasewidth] #0.15


    print('Going by planet')
    for i in range(nfiles):
        phases = []
        fluxes = []
        othertts = transtimes[:i] + transtimes[i+1:]
        if len(othertts) > 0:
            othertts = np.hstack(np.array(othertts,dtype=object))
        thistts = np.array(transtimes[i])
        for tti in thistts:
            if len(othertts) == 0:
                trange = np.where(np.abs(time - tti) < phasewidth[i])[0]
                phases.append(time[trange] - tti)
                fluxes.append(flux[trange])
            elif min(abs(othertts - tti)) > collisionwidth[i]:
                trange = np.where(np.abs(time - tti) < phasewidth[i])[0]
                try:
                    phases.append(time[trange] - tti)
                    fluxes.append(flux[trange])
                except AttributeError:
                    phases = np.append(phases, time[trange] - tti)
                    fluxes = np.append(fluxes, flux[trange])
                phases = np.hstack(phases)
                fluxes = np.hstack(fluxes)
                axes[i].scatter(phases, fluxes, s=0.01, c='gray', alpha=0.5)

        binwidth = 1./1440. * 10.
        nbins = int(2*phasewidth[i] / binwidth)
        binedges = np.arange(nbins+1, dtype=float)*binwidth - phasewidth[i]
        bincenters = np.arange(nbins, dtype=float)*binwidth - phasewidth[i] + binwidth/2.

        phasesort = np.argsort(phases)
        phases = phases[phasesort]
        fluxes = fluxes[phasesort]

        j=0
        k=0
        mbinned = np.zeros(nbins)
        while j < len(phases):
            mbinvals = []
            while phases[j] < binedges[k+1]:
                mbinvals.append(fluxes[j])
                j += 1
                if j >= len(phases):
                    break
            if len(mbinvals) > 0:
                mbinned[k] = np.mean(mbinvals)
            k += 1
            if k >= nbins:
                break

        axes[i].scatter(bincenters, mbinned, s=10., c=colorlist[i], label='Planet {}'.format(letters[i]))
        axes[i].set_xlim((-phasewidth[i], phasewidth[i]))
        axes[i].set_ylim((min(fluxes), max(fluxes)))
        axes[i].legend(fontsize=18)
        axes[i].set_ylabel('Flux', fontsize=20)

    plt.xlabel('Phase (days)', fontsize=20)
    #plt.ylabel('Normalized Flux')
    f.tight_layout()
    plt.savefig(outname+'.png')

Plot_Phasefold(time, flux, err, outname)


def Plot_all_transits(time, flux, error, model, outname):
    
    flux_start = np.min(time)
    flux_end = np.max(time)
    
    
    tbv = glob.glob("./tbv[0-9][0-9]_[0-9][0-9].out")
    for i in range(len(tbv)):
        f = tbv[i]
        ttimes = pd.read_csv(f, '  ', header=None)
        for j in range(len(ttimes[1])):
            
            if ttimes[1][j] < flux_start or ttimes[1][j] > flux_end:
                continue
            else:
                
                mask = (np.abs(time - ttimes[1][j]) < 0.3)
                
                
                try:
                    tran_min = np.min(flux[mask])
                    tran_max = np.max(flux[mask])
                except ValueError:
                    continue
            
                plt.figure(figsize=(8,3))

                plt.errorbar(time, flux, yerr=error, c='k', fmt='o') 
                plt.plot(time, model, c=colorlist[i], zorder=1000)
                plt.axvline(ttimes[1][j], ls='--', alpha=0.5, color='gray', label=np.round(ttimes[1][j],5))
                plt.title('Planet {}'.format(letters[i]))
                plt.legend()


                top = np.max(flux + 0.0001)

                plt.plot([ttimes[1][j], ttimes[1][j]], [top, top+1e-4],
                         c=colorlist[i], marker="None") 

                plt.xlabel('Time (days)', fontsize=20)
                plt.ylabel('Flux', fontsize=20)
                plt.xlim(ttimes[1][j] - 0.3, ttimes[1][j] + 0.3)
                plt.ylim(tran_min-1e-4, tran_max+1e-4)

                plt.savefig('Transits/{}_planet_{}_transit_{}.png'.format(outname, letters[i], ttimes[0][j]))

                plt.close()    



Plot_all_transits(time, flux, err, model, outname)
