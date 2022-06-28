#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the transmitted power for all mu in the half complex plane
at fixed nu.

Two mapping strategy are implement:
    - loop for mu at fixed nu
    - spanning the alpha_bar complex plane (using JSV formula)

@author: bn
"""
import matchingep as matching
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import concurrent.futures
# add progress bar for // loop
from tqdm import tqdm
import pickle

def get_power(admittances):
    """ Compute the powers.
    
    Parameters
    -----------
    admittances : tuple
        Contains the admittances of the two walls (mu, nu).
    """
    mu, nu = admittances
    print(k, mu, nu)
    Radius = (min(10, Nmodes) + 1.2)*np.pi
    # try:
    guide = matching.Guide(mu=mu,
                           nu=nu, k=k,
                           Nmodes=Nmodes, R=Radius)
    guide.matching(EP_order=EP)
    T1 = guide.transmitted_power(T_at_x)
    R = guide.reflected_power()
    # except:
        # R, T1 = 0, np.zeros_like(T_at_x)
    return R, T1


def loop(MU, nu, k, T_at_x):
    """ Compute the transmitted and the reflected power for k loop.

    Parameters
    -----------
    MU : 2D np.array
        The values of mu.
    NU : 2D np.array or complex
        The values of nu.
    """
    if not isinstance(nu, np.ndarray):
        NU = np.ones_like(MU, dtype=complex)*nu
    else:
        NU = nu

    T = np.zeros(MU.shape, dtype=object).ravel()
    R = np.zeros(MU.shape).ravel()

    # ProcessPoolExecutor = concurrent.futures.ProcessPoolExecutor
    # Number of split of each chunk to see the progress
    tqdm_each_n = 50
    # If not enougth data, just put one
    if MU.size//max_workers//tqdm_each_n < 1:
        tqdm_each_n = 1
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # use also from itertools import starmap to unpack
    for i, (Ri, Ti) in enumerate(tqdm(map(get_power,
                                          zip(MU.ravel(), NU.ravel())),
                                      total=NU.ravel().size)):
        R[i] = Ri
        T[i] = Ti

    T = T.reshape(MU.shape)
    R = R.reshape(MU.shape)
    return R, T


def reload(file):
    """Reload previous computation from the npz file `file`."""
    dat = np.load(file, allow_pickle=True)
    return dat['T'], dat['R'], dat['MU'], dat['k']


def alpha_bar_to_admittances(bounds, Nval):
    """Map the alpha_bar complex plane and generate the good admittances values."""
    def g(alpha):
        # numpy use sin(pi x)/(pi x)
        return np.sinc(2*alpha / np.pi)

    # Mesh alpha_bar complex plane
    alphar, alphai = np.meshgrid(np.linspace(bounds[0].real, bounds[1].real, Nval),
                                 np.linspace(bounds[0].imag, bounds[1].imag, Nval))
    alpha = alphar + 1j * alphai

    p = alpha**2 * (1 + g(alpha)) / (1 - g(alpha))
    s = - np.tan(alpha)*(alpha - p/alpha)
    nu = (s + np.sqrt(s**2 - 4*p, dtype=complex))/2
    mu = (s - np.sqrt(s**2 - 4*p, dtype=complex))/2
    return alpha, mu, nu


def pickle_fig(name, fig_handle):
    """Save figure handle to disk."""
    with open(name, 'wb') as f:
        pickle.dump(fig_handle, f)


def load_fig(name):
    """Reload a figure from a pickel file."""
    with open(name, 'rb') as f:
        p = pickle.load(f)
    return p


def plot_T(T_at_x, T, alpha_bar, passive, alpha_3, k):
    """Plot the |T|^2 map."""
    T_ = np.zeros((*alpha_bar.shape, T_at_x.size))
    for (i, j), mu in np.ndenumerate(alpha_bar):
        T_[i, j, :] = T[i, j]
    # Nlevel = 8
    cmap = cm.get_cmap('viridis_r')
    for i, x in enumerate(T_at_x):
        figmu, axmu = plt.subplots(num='log10 T x=' + str(x) + 'at k=' + str(k))
        # axmu.set_title(str(x))
        T_[np.invert(passive)] = np.nan
        # remove non physical value for passive
        ind = np.nonzero(T_ > 1)
        T_[ind] = np.nan
        pc = axmu.contourf(alpha_bar.real, alpha_bar.imag, np.log10(T_[:, :, i]),
                           cmap=cmap, vmax=0, levels=16)
        # pc = axmu.pcolormesh(alpha_bar.real, alpha_bar.imag, np.log10(T_[:,:, i]), cmap=cmap)
        figmu.colorbar(pc)
        axmu.set_xlabel(r'Re $\bar{\alpha}$')
        axmu.set_ylabel(r'Im $\bar{\alpha}$')
        axmu.plot(alpha_3.real, alpha_3.imag, 'k*', zorder=5)


def plot_R(R, alpha_bar, passive, alpha_3, k):
    """Plot |R|^2 map."""
    Nlevel = 13
    cmapR = cm.get_cmap('viridis_r')
    figmuR, axmuR = plt.subplots(num='R at k=' + str(k))
    R_ = R.copy()
    R_[np.invert(passive)] = np.nan
    Rmin, Rmax = 0, .05
    Rlevels = np.linspace(Rmin, Rmax, Nlevel)
    pcR = axmuR.contourf(alpha_bar.real, alpha_bar.imag, -R_, Rlevels, vmin=Rmin, vmax=Rmax, cmap=cmapR)
    cbar = figmuR.colorbar(pcR)
    axmuR.set_xlabel(r'Re $\bar{\alpha}$')
    axmuR.set_ylabel(r'Im $\bar{\alpha}$')
    axmuR.plot(alpha_3.real, alpha_3.imag, 'k*', zorder=5)


def replot(filename):
    """Replot figure from saved data."""
    dat = np.load(filename, allow_pickle=True)
    R, T, k, passive, alpha_3 = dat['R'], dat['T'], dat['k'], dat['passive'], dat['alpha_3']
    MU, NU = dat['MU'], dat['NU']
    # I forget to save alpha_bar!
    try:
        alpha_bar = dat['alpha_bar']
    except:
        alpha_bar, _, _ = alpha_bar_to_admittances(alpha_bar_bounds, MU.shape[0])
    plot_R(R, alpha_bar, passive, alpha_3, k)
    plot_T(T_at_x, T, alpha_bar, passive, alpha_3, k)


if __name__ == '__main__':
    # %% Run options
    plt.close('all')
    k = 5
    Nmodes = 30
    max_workers = 4
    T_at_x = np.array([10., 50, 100])
    Nval = 35

    EP_k0 = 3
    filename = './map_alpha_bar/test_' + str(k)
    file_replot = 'map_alpha_bar/test_k=5.npz'  # used for replot strategy
    # reference EP3 value
    mu0 = (3.08753629148967+3.62341792246081j)
    nu0 = (3.17816250726595+4.67518038763374j)
    EPref = 3
    # Reference EP2 Tester
    # mu0 = (1.650611293539765+2.059981457179885j)
    # nu0 = 0
    # EPref = 2
    nu = nu0
    strategy = 'alpha_bar'  # in {'alpha_bar', 'nu_fixed', 'replot'}
    alpha_bar_bounds = (1e-2-4j, 12-1j)
    # alpha_bar_bounds = (2-3j, 6-2j)
    # %% 'nu_fixed' strategy
    if strategy == 'nu_fixed':
        # Span simple roots
        EP = 0
        mur, mui = np.meshgrid(np.linspace(-5, 7, Nval),
                               np.linspace(0.1, 7, Nval))
        MU = mur + 1j * mui
        # Compute for many admittance values
        R, T = loop(MU, nu0, k, T_at_x)

        # %% Plot
        figr, axr = plt.subplots()
        dmax = np.abs(MU-mu0).max()
        T_ = np.zeros((*MU.shape, T_at_x.size))
        for (i, j), mu in np.ndenumerate(MU):
            T_[i, j, :] = T[i, j]

        # Use distance
        for i, x in enumerate(T_at_x):
            figmu, axmu = plt.subplots()
            axmu.set_title(str(x))
            pc = axmu.contourf(MU.real, MU.imag, np.log10(T_[:, :, i]), vmax=1)
            figmu.colorbar(pc)
            axmu.set_xlabel(r'Re $\mu$')
            axmu.set_ylabel(r'Im $\mu$')
            axmu.plot(mu0.real, mu0.imag, 'k*', zorder=5)

        # %% R
        figmuR, axmuR = plt.subplots()
        # %pcR = axmuR.contourf(MU.real, MU.imag, np.log10(-R))
        pcR = axmuR.contourf(MU.real, MU.imag, -R)  # , vmin=0, vmax=.1
        figmuR.colorbar(pcR)
        axmuR.set_xlabel(r'Re $\mu$')
        axmuR.set_ylabel(r'Im $\mu$')
        axmuR.plot(mu0.real, mu0.imag, 'k*', zorder=5)

        np.savez(filename, T=T, R=R, MU=MU, NU=nu0, k=k)

    # %% 'alpha_bar' strategy
    elif strategy == 'alpha_bar':
        # span all double roots
        EP = 2
        alpha_bar, MU, NU = alpha_bar_to_admittances(alpha_bar_bounds, Nval)
        # Compute for many admittance values
        R, T = loop(MU, NU, k, T_at_x)
        # Compute it also for nominal value
        guide = matching.Guide(mu=mu0,
                               nu=nu0, k=k,
                               Nmodes=Nmodes, R=10)
        guide.matching_EP3()
        T3 = guide.transmitted_power(T_at_x)
        R3 = guide.reflected_power()
        # Compute passive
        passive = (MU.imag > 0) & (NU.imag > 0)
        alpha_3 = guide.alpha[guide.get_index_at_multiplicity(3)]
        # plot
        plot_T(T_at_x, T, alpha_bar, passive, alpha_3, k)

        plot_R(R, alpha_bar, passive, alpha_3, k)
        # save npz
        np.savez_compressed(filename, T=T, R=R, MU=MU, NU=NU, k=k, T3=T3, R3=R3,
                            passive=passive, alpha_3=alpha_3, alpha_bar=alpha_bar)
        # %% pickle all figures
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pickle_fig(fig.get_label() + '.pickle', fig)
    # %% replot strategy
    elif strategy == 'replot':
        replot(file_replot)
