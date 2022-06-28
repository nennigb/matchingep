#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frequency loop.

@author: bn
"""
import matchingep as matching
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import concurrent.futures
# %% Main

def get_reflected_power(k):
    R = (min(10, Nmodes) + 1.2)*np.pi
    guide = matching.Guide(mu=mu,
                           nu=nu, k=k,
                           Nmodes=Nmodes, R=R)
    if EP == 3:
        guide.matching_EP3()
    if EP == 2:
        guide.matching_EP2()
    if EP == 0:
        guide.matching_std()
        mult = np.round(guide.multiplicities.real)
        if (mult > 1).any():
            print('WARNING > Some roots have multiplicties > 1. Try an other matching strategy.')
    T1 = guide.transmitted_power(T_at_x)
    return guide.reflected_power(), guide.A, T1


def broken_axis(x, y, ax1_ylim=(30, 40), ax2_ylim=(0, 2)):
    # If we were to simply plot pts, we'd lose most of the interesting
    # details due to the outliers. So let's 'break' or 'cut-out' the y-axis
    # into two portions - use the top (ax1) for the outliers, and the bottom
    # (ax2) for the details of the majority of our data
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 3]})
    fig.subplots_adjust(hspace=0.05)  # adjust space between axes

    # plot the same data on both axes
    ax1.plot(x, y, 'k')
    ax2.plot(x, y, 'k')

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(ax1_ylim)  # outliers only
    ax2.set_ylim(ax2_ylim)  # most of the data

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    # ax1.xaxis.set_ticks([])
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    return fig, ax1, ax2


def pi_ticks(x):
    """Return the ticks position and label for pi multiple."""
    ntick = int(x[-1]/np.pi+1)
    ticks = []
    xticks = np.arange(0, ntick, 0.5)*np.pi
    for i, _ in enumerate(xticks):
        if i == 0:
            ticks.append('$0$')
        elif i == 2:
            ticks.append('$\\pi$')
        elif (i >= 1) and (i % 2 == 0):
            ticks.append('$' + str(i//2) + '\\pi$')
        elif (i >= 1) and (i % 2 == 1):
            ticks.append(r'$ \frac{{{}}}{{2}}\pi$'.format(i))
    return xticks, ticks



if __name__ == '__main__':
    # %% Run options

    k_vec = np.linspace(1, 7.85, 250)
    # k_vec = np.array([4.])
    color = 'b'
    Nmodes = 50
    max_workers = 4
    T_at_x = 5
    save = False
    # EP = 3
    # mu = 3.08753629148967+3.62341792246081j
    # nu = 3.17816250726595+4.67518038763374j
    # coef = 1.

    # EP3 PT 1
    EP = 3
    coef = 1
    mu = (1.0119407382877632+4.602904394703379j)*coef
    nu = (1.0119407382877632-4.602904394703379j)*coef

    # EP3 PT (7.789350443630617+1.5654800548642723e-13j)
    # EP = 3
    # coef = 1
    # mu = (1.0041371457463308-7.7896171019767975j)
    # nu = (1.0041371457463308+7.7896171019767975j)

    # EP3 higher
    # EP = 3
    # coef = 1
    # mu = (3.6015590344615322+6.945949924732483j)*coef
    # nu = (3.659876418200885+7.968433151998626j)*coef

    # Standard matching with 10%, 1%, 0.1% from admittance EP value
    # EP = 0
    # coef = 0.99999
    # mu = (3.08753629148967+3.62341792246081j)*coef
    # nu = (3.17816250726595+4.67518038763374j)*coef

    # EP = 0
    # mu=3.08+3.j
    # nu = 3.17+4j

    # EP = 0
    # mu=0.2j
    # nu = 0.1j

    # EP2
    # EP = 2
    # mu = 0.
    # nu = 1.650611293539765+2.059981457179885j

    # EP = 0
    # mu = 0.1+0.3j
    # nu = mu.conjugate()
    # coef = 1.
    # plt.close('all')
    filename = 'temp/RefP{}_mu{}_nu{}_coeff_{}'.format(EP, mu, nu, coef)
    refp = np.zeros_like(k_vec)
    transp = np.zeros_like(k_vec)
    A = np.zeros((Nmodes, k_vec.size), dtype=complex)
    Nk = k_vec.size

    ProcessPoolExecutor = concurrent.futures.ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, (R, Ai, T1) in enumerate(executor.map(get_reflected_power,
                                                     k_vec, chunksize=k_vec.size//max_workers)):
            print("\ni = {}  - |R|²={}".format(i, R))
            refp[i] = R
            transp[i] = T1
            A[:, i] = Ai

# %% plot
fig = plt.figure()
ax = plt.subplot()
# by convention -r
plt.plot(k_vec, -refp, 'k')
plt.xlabel('$k$')
plt.ylabel('$\mathcal{R}(k)$')
ylim = list(plt.ylim())
ylim[0] = 0
R = (min(10, Nmodes) + 1.2)*np.pi
guide = matching.Guide(mu=mu,
                       nu=nu, k=1,
                       Nmodes=Nmodes, R=R)

# %% linear scale
plt.vlines(np.pi, *ylim, linestyle=':', color='grey')
plt.vlines(2*np.pi, *ylim, linestyle=':', color='grey')
plt.vlines(guide.alpha[0], *ylim, linestyle=':', color='b')
if guide.alpha[1].real < k_vec[-1]:
    plt.vlines(guide.alpha[1], *ylim, linestyle=':', color='b')
ax.set_ylim(ylim)
# broken axis version
# fig, ax1, ax2 = broken_axis(k_vec, -refp, ax1_ylim=(5, ylim[1]), ax2_ylim=(0, 1.25))
# plt.vlines(np.pi, *ylim, linestyle=':', color='grey')
# plt.vlines(2*np.pi, *ylim, linestyle=':', color='grey')
# plt.vlines(guide.alpha[0], *ylim, linestyle=':', color='b')
# plt.xlabel('$k$')
# plt.ylabel(r'$\mathcal{R}(k)$', loc='top')

# %% log scale
figlog = plt.figure()
plt.semilogy(k_vec, -refp, 'k')
plt.ylim(1e-3, max(100, np.abs(refp).max()))
yliml = list(plt.ylim())
plt.vlines(np.pi, *yliml, linestyle=':', color='grey')
plt.vlines(2*np.pi, *yliml, linestyle=':', color='grey')
plt.vlines(guide.alpha[0], *yliml, linestyle=':', color='b')
xticks, ticks = pi_ticks(k_vec)
plt.xticks(xticks, ticks)
plt.xlabel(r'$k$')
plt.ylabel(r'$\mathcal{R}(k)$')

fig_coef = plt.figure()
plt.pcolormesh(k_vec, np.arange(0, Nmodes), np.abs(A))
plt.xlabel('$k$')
plt.ylabel('$A_n$')

# %% log scale P2
figTlog = plt.figure('Transmitted power at x={}'.format(T_at_x))
plt.plot(k_vec, transp, 'k')
yliml = list(plt.ylim())
plt.vlines(np.pi, *yliml, linestyle=':', color='grey')
plt.vlines(2*np.pi, *yliml, linestyle=':', color='grey')
plt.vlines(guide.alpha[0], *yliml, linestyle=':', color='b')
plt.xticks(xticks, ticks)
plt.xlabel(r'$\frac{k}{\pi}$')
plt.ylabel(r'$\mathcal{T}(k)$')


# %% save data
if save:
    np.savez(filename, k_vec, mu=mu, nu=nu, refp=refp, A=A)