#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frequency loop to compute TL of a 1DOF liner.


@author: bn
"""
import matchingep as matching
import numpy as np
from matplotlib import pyplot as plt
import concurrent.futures
from functools import partial


def get_powers(k, mu, nu, EP):
    """Compute reflected and transmitted power."""
    R = (min(10, Nmodes) + 1.2)*np.pi
    guide = matching.Guide(mu=mu.eval_at(k),
                           nu=nu.eval_at(k), k=k,
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
    return guide.reflected_power(), T1


class Liner_1DOF:
    """Define a 1DOF liner.

    The definition is not unique and can be adjusted with the `d` parameter.

    Examples
    --------
    >>> d = 0.75
    >>> mu = Liner_1DOF(R=0.1598893248570865, b=0.13624265919145442 + d, d=d)
    >>> abs(mu.eval_at(1.) - (3.08753629148967+3.62341792246081j)) < 1e-12
    True

    >>> mu2 = Liner_1DOF.from_EP3(1., (3.08753629148967+3.62341792246081j))
    >>> abs(mu.eval_at(1.) - mu2.eval_at(1.)) < 1e-12
    True
    """

    def __init__(self, R, b, d, zero=False):
        self.R = R
        self.b = b
        self.d = d
        self.zero = zero

    def __repr__(self):
        """Define the custom representation."""
        return "Instance of {} with R={}, b={} and d={}.".format(self.__class__.__name__,
                                                                 self.R,
                                                                 self.b,
                                                                 self.d)

    def eval_at(self, k):
        """Compute the admittance value."""
        if self.zero:
            admittance = 0.
        else:
            admittance = 1j*k**2 / (self.R*k + 1j*self.b - 1j*self.d*k**2)
        return admittance

    def get_pole(self):
        """Compute roots of the denominator."""
        return np.roots([-1j*self.d, self.R, 1j*self.b])

    @classmethod
    def from_EP3(cls, k0, mu3, d=0.75):
        """Compute R, b and d from EP3 values at k0."""
        R = (1j / mu3).real
        b = (1j / mu3).imag + d
        return cls(R, b, d)


def test():
    """Run test suite."""
    import doctest
    doctest.testmod()


# %% Main
if __name__ == '__main__':
    from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                      mark_inset, zoomed_inset_axes)
    # %% Run options
    case = 'EP3'  # {'EP2_Tester', 'EP3'}
    # x values where T is compute
    T_at_x = [1, 5]
    # create k vec passing through k0
    N = 1000
    k0 = 1.
    shift = 0
    k_vec = np.linspace(0.1, 3.5, N)
    # k_vec = np.linspace(0.9, 1.5, N)
    idx = (np.abs(k_vec - k0)).argmin()
    k_vec[idx] = k0 + shift
    # k_vec = np.array([4.])
    color = 'b'
    Nmodes = 30
    EP = 0
    inset = True
    max_workers = 4
    R = (min(10, Nmodes) + 1.2)*np.pi
    d = 0.75
    # Define setup for EP3 or EP2 Tester
    if case == 'EP3':
        mu = Liner_1DOF(R=0.1598893248570865,
                        b=0.13624265919145442 + d,
                        d=d)
        nu = Liner_1DOF(R=0.1462912637430899,
                        b=0.0994480150538454 + d,
                        d=d)

        # EP3
        mu3 = 3.08753629148967+3.62341792246081j
        nu3 = 3.17816250726595+4.67518038763374j
        # Compute at the exact value of the EP3
        guide = matching.Guide(mu=mu.eval_at(k0),
                               nu=nu.eval_at(k0), k=k0,
                               Nmodes=Nmodes, R=R)
        guide.matching_EP3()
    elif case == 'EP2_Tester':
        mu = Liner_1DOF(R=None, b=None, d=None, zero=True)
        nu = Liner_1DOF(R=0.2956327875179946,
                        b=0.23688311179552127 + d,
                        d=d)
        guide = matching.Guide(mu=mu.eval_at(k0),
                               nu=nu.eval_at(k0), k=k0,
                               Nmodes=Nmodes, R=R)
        guide.matching_EP2()
    refp = np.zeros_like(k_vec)
    Nk = k_vec.size
    transp = np.zeros((Nk, len(T_at_x)), dtype=float)

    # %% Loop
    ProcessPoolExecutor = concurrent.futures.ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        partial
        get_prower_k = partial(get_powers, mu=mu, nu=nu, EP=EP)
        for i, (R, T) in enumerate(executor.map(get_prower_k, k_vec, chunksize=k_vec.size//max_workers)):
            print("\ni = {}  - |R|²={}, TL={}".format(i, R, np.log10(T)))
            refp[i] = R
            transp[i, :] = T

    R3 = guide.reflected_power()
    T3 = guide.transmitted_power(T_at_x)
# %% plot
fig = plt.figure('R')
ax = plt.subplot()
# by convention -r
ax.plot(k_vec, -refp, 'k')
ax.plot(k0, -R3, 'k*')
ax.set_xlabel('$k$')
ax.set_ylabel(r'$\mathcal{R}$')

fig = plt.figure('T')
axT = plt.subplot()
color = ['k', 'gray', 'darkgray', 'lightgray']

if inset:
    zoom = 7
    axT_ins = zoomed_inset_axes(axT, zoom, loc=7, borderpad=5)  # 1
    # draw a bbox of the region of the inset axes
    mark_inset(axT, axT_ins, loc1=2, loc2=4, fc="none", ec="0.75")
    axT_ins.set_xlim(0.95, 1.05)  # Limit the region for zoom
    if case == 'EP3':
        axT_ins.set_ylim(2.5*10, 3.8*10) # Limit the region for zoom
    elif case == 'EP2_Tester':
        axT_ins.set_ylim(10, 18)  # Limit the region for zoom

ax_list = [axT, axT_ins]
for ax in ax_list:
    for i, x in enumerate(T_at_x):
        ax.plot(k_vec, -10*np.log10(transp[:, i]), label=r'x={}'.format(x), color=color[i], marker='')
        ax.plot(k0, -10*np.log10(T3[i]), '*', color=color[i])

axT.set_xlabel('$k$')
axT.set_ylabel(r'TL (dB)')
axT.legend()
