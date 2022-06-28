#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replot `convergence.py` results.
Use to produce paper figures.

Created on Mon Sep 13 15:38:53 2021

@author: bn
"""
import numpy as np
from matplotlib import pyplot as plt
from convergence import conv_plot, reload_file, find_slope, cond_plot, Refp_plot

plt.close('all')
# Files list
fileEP3 = 'res/EP3_k1.0_mu(3.08753629148967+3.62341792246081j)_nu(3.17816250726595+4.67518038763374j)_coeff_1.0.npz'
fileEP0 = 'res/EP0_k1.0_mu(3.08444875519818+3.619794504538349j)_nu(3.174984344758684+4.670505207246107j)_coeff_0.999.npz'
fileEP00 = 'res/EP0_k1.0_mu(3.087227537860521+3.623055580668564j)_nu(3.1778446910152236+4.674712869594977j)_coeff_0.9999.npz'
#fileEP00 = 'res/EP0_k1.0_mu(3.087505416126755+3.6233816882815852j)_nu(3.1781307256408775+4.675133635829864j)_coeff_0.99999_.npz'

# EP3 enhance matching
tag = 'EP3'
Nmodes_vec, Err2_Psi, Err2_v, reflected_power, Cond2 = reload_file(fileEP3)
for_psi3 = conv_plot(Nmodes_vec, Err2_Psi, color='k', marker='.', name=r'\psi', fit=True, tag=tag)
for_v3 = conv_plot(Nmodes_vec, Err2_v, color='k', marker='.', name=r'{\partial_x\psi}', fit=True, tag=tag)
for_cond3 = cond_plot(Nmodes_vec, Cond2, color='k', marker='.', tag=tag)
for_ref = Refp_plot(Nmodes_vec, reflected_power, color='b', marker='.', tag=tag, ax=None)

# EP0
tag = r'Standard ($10^{-3}$)'
Nmodes_vec, Err2_Psi, Err2_v, reflected_power, Cond2 = reload_file(fileEP0)
for_psi0 = conv_plot(Nmodes_vec, Err2_Psi, color='b', marker='+',
                     name=r'\psi', fit=False, tag=tag, ax=for_psi3)
for_v0 = conv_plot(Nmodes_vec, Err2_v, color='b', marker='+',
                   name=r'{\partial_x\psi}', fit=False, tag=tag, ax=for_v3)

cond = cond_plot(Nmodes_vec, Cond2, color='b', marker='+', tag=tag, ax=for_cond3)
ref = Refp_plot(Nmodes_vec, reflected_power, color='b', marker='+', tag=tag, ax=for_ref)

# EP00
tag = r'Standard ($10^{-4}$)'
marker = 'x'
Nmodes_vec, Err2_Psi, Err2_v, reflected_power, Cond2 = reload_file(fileEP00)
for_psi0 = conv_plot(Nmodes_vec, Err2_Psi, color='b', marker=marker,
                     name=r'\psi', fit=False, tag=tag, ax=for_psi3)
for_v0 = conv_plot(Nmodes_vec, Err2_v, color='b', marker=marker,
                   name=r'{\partial_x\psi}', fit=False, tag=tag, ax=for_v3)

cond = cond_plot(Nmodes_vec, Cond2, color='b', marker=marker, tag=tag, ax=for_cond3)
ref = Refp_plot(Nmodes_vec, reflected_power, color='b', marker=marker, tag=tag, ax=for_ref)
