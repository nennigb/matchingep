#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute all triple roots. Use equations defined in
10.1016/j.jsv.2021.116510

Created on Tue Sep 14 13:56:14 2021

@author: bn
"""
import numpy as np
from matplotlib import pyplot as plt
import polze


if __name__ == '__main__':
    Ni = 30000
    Npz = 18
    R = 12
    affich = True
    refine = False

    def K3(z):
        """Equation satisfied by triple roots."""
        return 4.*np.cos(z)*z**2 - np.sin(z) * (2 * z + np.sin(2*z))

    pz = polze.PZ(K3, Rmax=R, Npz=Npz, Ni=Ni, split=True,
                  options={'_vectorized': True, '_Npz_limit': 4,
                           '_zeros_only': True,
                           '_tol': 1e-4})
    pz.solve()
    _, (alphan, multiplicity) = pz.dispatch(refine=refine, multiplicities=True)
    pos = alphan.real > 0
    alphan = alphan[pos]
    multiplicity = multiplicity[pos]

    p = alphan**2 * ((2*alphan + np.sin(2.*alphan))
                     / (2*alphan - np.sin(2.*alphan)))
    s = - np.tan(alphan)*(alphan - p/alphan)
    # l'admintance Y optimale
    nus = (s + np.sqrt(s**2 - 4*p, dtype=complex))/2.
    mus = (s - np.sqrt(s**2 - 4*p, dtype=complex))/2.
    space = 17*' '
    tag = ['alpha', '|', 'mu', '|', 'nu']
    print(space + space.join(tag))
    for alpha, mu, nu in zip(alphan, mus, nus):
        print(alpha, mu, nu)

    if affich:
        def get_color(alpha):
            """Provide color according to the imaginary part sign."""
            tol = 1e-5
            if alpha.imag > tol:
                color = 'r'
            elif alpha.imag < -tol:
                color = 'b'
            else:
                color = '0.5'
            return color

        alpha2_tester = 2.10619612-1.12536431j
        # %% Plot the wavenumber
        plt.figure('alpha_plane')
        plt.plot(alpha2_tester.real, alpha2_tester.imag, '+', color='b', markersize=6)
        alpha_used = np.array([alpha2_tester, 4.19693888-2.60864154e+00j, 4.60159163])
        labels = ['EP2', 'EP3', 'EP3-PT']
        plt.plot(alpha_used.real, alpha_used.imag, 'o', markersize=8,
                 markeredgecolor='0', markerfacecolor='none')
        for n, alpha in enumerate(alphan):
            color = get_color(alpha)
            plt.plot(alpha.real, alpha.imag, '^', color=color,  markersize=4)
        for n, (alpha, label) in enumerate(zip(alpha_used, labels)):
            plt.text(alpha.real + 0.1, alpha.imag + 0.1, label)
        plt.xlabel(r'Re $\alpha$')
        plt.ylabel(r'Im $\alpha$')
        plt.xticks([0, np.pi, 2*np.pi, 3*np.pi], ['0', r'$\pi$', r'2$\pi$', r'3$\pi$'])
        plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                   [r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])
