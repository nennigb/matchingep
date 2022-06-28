#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute convergence curves for the different matching strategies.

Created on Wed Sep  1 20:28:43 2021

@author: bn
"""
import matchingep as matching
import numpy as np
from matplotlib import pyplot as plt


# %% plotting helpers
def find_slope(b, x):
    """Use LMS to estimate slope of data."""
    A = np.block([[np.ones_like(x)], [x]]).T
    p = np.linalg.pinv(A) @ b
    return p


def reload_file(outfile):
    """Reload data from file."""
    data = np.load(outfile)
    return data['Nmodes'], data['Err2_Psi'], data['Err2_v'], \
        data['reflected_power'], data['Cond2']


def conv_plot(Nmodes_vec, Err2, color='k', marker='.',
              name=r'\psi', fit=True, tag='standard', ax=None):
    """Plot convergence curves."""
    if ax is None:
        fig = plt.figure(r'Convergence for ' + name)
        ax = fig.add_subplot(111)
    ax.loglog(Nmodes_vec, Err2, color + marker, label=tag)
    # Fitting
    if fit:
        p = find_slope(np.log10(Err2), np.log10(Nmodes_vec))
        ax.plot(Nmodes_vec, (10**p[0])*Nmodes_vec**(p[1]),
                color=color, linestyle='--', label='LMS fit')
        print('Fitted p=', p)
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'$\mathcal{{E}}^{}_N$'.format(name))
    ax.legend()
    # ax.grid()
    return ax


def cond_plot(Nmodes_vec, Cond2, color='k', marker='.', tag='standard', ax=None):
    """Plot condition number curve."""
    if ax is None:
        fig = plt.figure(r'Cond_2 ' + tag)
        ax = fig.add_subplot(111)
    ax.semilogy(Nmodes_vec, Cond2, color + ':' + marker, label=tag)
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'Condition Number')
    ax.legend(loc='upper right')
    # ax.grid()
    return ax


def Refp_plot(Nmodes_vec, ref, color='k', marker='.', tag='standard', ax=None):
    """Plot condition number curve."""
    if ax is None:
        fig = plt.figure(r'R ' + tag)
        ax = fig.add_subplot(111)
    ax.plot(Nmodes_vec, ref, color + ':' + marker, label=tag)
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'Reflected power')
    ax.legend(loc='lower right')
    # ax.grid()
    return ax


# %% Main
if __name__ == '__main__':
    # %% Run options
    filter_name = 'None'
    k = 1.
    color = 'b'
    fit = True

    # EP3
    EP = 3
    mu = 3.08753629148967+3.62341792246081j
    nu = 3.17816250726595+4.67518038763374j
    coef = 1.

    # EP3 PT
    # EP = 3
    # coef = 1
    # mu = (1.0119407382877632+4.602904394703379j)*coef
    # nu = (1.0119407382877632-4.602904394703379j)*coef

    # EP3 higher
    # EP = 3
    # coef = 1
    # mu = (3.6015590344615322+6.945949924732483j)*coef
    # nu = (3.659876418200885+7.968433151998626j)*coef

    # EP3 PT (7.789350443630617+1.5654800548642723e-13j)
    # EP = 0
    # coef = 0.999
    # mu = 0.1
    # nu = 0.1j

    # Standard matching with 10%, 1%, 0.1% from admittance EP value
    # EP = 0
    # coef = 0.99999
    # mu = (3.08753629148967+3.62341792246081j)*coef
    # nu = (3.17816250726595+4.67518038763374j)*coef

    # EP2 Tester
    # EP = 2
    # coef = 1
    # mu = 0.
    # nu = 1.650611293539765+2.059981457179885j

    # EP = 0
    # coef = 0.999
    # mu = 0.
    # nu = (1.650611293539765+2.059981457179885j)*coef

    Nmodes_vec = np.logspace(1, 3, 25).astype(int)
    # Nmodes_vec = np.array([10, 20, 50, 100, 200])
    Err2_Psi = np.zeros_like(Nmodes_vec, dtype=float)
    Err2_v = np.zeros_like(Nmodes_vec, dtype=float)
    reflected_power = np.zeros_like(Nmodes_vec, dtype=float)
    Cond2 = np.zeros_like(Nmodes_vec, dtype=float)

    # output file name
    outfile = 'res/EP{}_k{}_mu{}_nu{}_coeff_{}'.format(EP, k, mu, nu, coef)

    # Convergence loop
    for i, Nmodes in enumerate(Nmodes_vec):
        print("\nNmodes = %d" % Nmodes)
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
                print('Warning some roots have multiplicties > 1.')
        # print(guide.transmitted_power(np.array([1., 2.])))
        #  guide.filter_coefs(filter_name)
        Err2_Psi_, Err2_v_ = guide.get_L2_error()
        Err2_Psi[i] = Err2_Psi_
        Err2_v[i] = Err2_v_
        reflected_power[i] = guide.reflected_power()
        Cond2[i] = guide.get_cond2()

    # %% Save
    np.savez(outfile, Nmodes=Nmodes_vec, Err2_Psi=Err2_Psi, Err2_v=Err2_v,
             reflected_power=reflected_power, Cond2=Cond2, mu=mu, nu=nu, EP=EP)

    # %% Plotting
    conv_psi = conv_plot(Nmodes_vec, Err2_Psi, color='k', marker='.',
                         name=r'\psi', fit=True, tag=guide._matching_method)
    conv_v = conv_plot(Nmodes_vec, Err2_v, color='k', marker='.',
                       name=r'{\partial_x\psi}', fit=True, tag=guide._matching_method)
    cond = cond_plot(Nmodes_vec, Cond2, color='k', marker='.', tag=guide._matching_method)

    plt.figure('Reflected power')
    plt.plot(Nmodes_vec, reflected_power, color+':.')
    plt.xlabel('N')
    plt.ylabel(r'$R$')
