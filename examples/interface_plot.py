#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots interface fields and maps of the potential along the waveguide.

Created on Tue Sep 14 13:56:14 2021

@author: bn
"""
import copy
import matchingep as matching
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # frequency
    k = 1

    Nmodes = 60
    filtered = False
    inset = True
    filter_name = 'None'
    R = (min(50, Nmodes) + 2)*3.14
    # 1st lossy EP
    EP = 3
    mu = 3.08753629148967+3.62341792246081j
    nu = 3.17816250726595+4.67518038763374j

    # EP = 0
    # mu = 1j
    # nu = 1.17816250726595+5.67518038763374j

    # 1st PT EP
    # coef = 1
    # mu = (1.0119407382877632+4.602904394703379j)*coef
    # nu = (1.0119407382877632-4.602904394703379j)*coef

    # EP2 Tester
    # EP = 2
    # coef = 1
    # mu = 0.
    # nu = 1.650611293539765+2.059981457179885j

    guide = matching.Guide(mu=mu, nu=nu, k=k, Nmodes=Nmodes, R=R)
    # Compute the matching
    guide.matching(EP_order=EP)

    if EP > 0:
        n = np.where(np.round(guide.multiplicities) == EP)[0].item()
        print('Err on K(alpha1) and Kp(alpha1)', abs(guide._K(guide.alpha[n])),
              abs(guide._Kp(guide.alpha[n])),
              abs(guide._Kpp(guide.alpha[n])))

    Err2_Psi, Err2_v = guide.get_L2_error()
    # Raw case
    ax1, ax2 = guide.plot_field(Ny=600, markerevery=10, inset=inset)
    # Filtered case
    if filtered:
        guide.filter_coefs(filter_name)
        guide.plot_field(Ny=600, markerevery=10)
    # Show that additional function are mandatory in the reconstruction
    # guide_ = copy.deepcopy(guide)
    # guide_.B[-1::] = 0
    # guide_.plot_field(Ny=200)
    # guide.B[-1::] = 0
    guide.plot_field(only_interface=False, Nx=500, Ny=500)
