#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# matchingep : A python package for extending mode matching method on 2D
# waveguide tuned at an exceptional point.
# Copyright (C) 2021  B. Nennig (benoit.nennig@isae-supmeca.fr)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Main module.
"""
from abc import ABC, abstractmethod
import numpy as np
import polze
from matplotlib import pyplot as plt
import scipy.optimize as spop
from scipy.integrate import simpson
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
cos, sin, exp = np.cos, np.sin, np.exp


class Guide():
    """Define the Guide class.

    The guide is a semi-infinite waveguide with a rigid duct on the left side
    (x<0) and a lined duct (x>0) on the right side
    ```txt
                                     nu
    ====================~~~~~~~~~~~~~~~~~~~~~~~
         F              :       B_n
       |-|->            :      |-|->
              A_n <-|-| :
    ====================~~~~~~~~~~~~~~~~~~~~~~~
         wall          x=0           mu
    ```

    `mu` and `nu` denotes the two wall admittances respectivelly on the bottom
    (y=0) and on the top (y=1).

    All overlap integrals are delegated to the `Product` class.
    """

    _skiptol = 1e-9             # Use to skip alpha imaginary part
    _alpha_pos_tol = - _skiptol  # Use to select only real > 0 alpha

    def __init__(self, mu=1, nu=1j, k=1, Nmodes=2, R=5*3.14, alpha=None, mult=None):
        """Initialize the Guide object instance.

        Parameters
        ----------
        mu : complex
            The bottom admittances.
        nu : complex
            Top admittances.
        k : float
            The freefield wave number.
        Nmodes : int
            The number of modes used in the expansion.
        R : float, optional
            The radius used in contour solver for finding the first roots.
            Higher order roots are obtained from assymptotics and Newton-Raphson
            method. The default is 5*3.14.
        alpha : array, optional
            Initialize the instance with the transverse wavenumber. It avoids to
            recompute them. Usefull for repeated computation with the same
            admittance (eg chnaging `k` only). The default is None.
        mult : array, optional
            Initialize the instance with the transverse wavenumber multiplicities.
            It avoids to recompute the roots. The default is None. Should be
            used with `alpha`.
        """

        self.a = 0.
        self.b = 1.  # not safe to change it.
        self.mu = mu
        self.nu = nu
        self.k = k
        # Number of mode in the rigid part
        self.Nmodes = Nmodes
        if alpha is None:
            # solve method set self.alpha and self.multiplicities
            alpha = self.solve_disp(R)
        else:
            self.alpha = alpha
            self.multiplicities = mult
        # Check if the imaginary part of the transverse wavenumber is nearly 0.
        # If True, the imaginary part is skipped to avoid jump due to the
        # branch cut crossing in axial wavenumber computation
        # This situation is expected when the admittance are complex conjugate
        # if (np.abs(self.alpha.imag) < self._skiptol).all():
        #     print('INFO    > Skip `alpha` imaginary part '
        #           +' (tol={}). '.format(self._skiptol)
        #           + 'Max(Im alpha) = {}.'.format(np.abs(self.alpha.imag).max()))
        #     self.alpha = self.alpha.real + 0j
        indices = np.where(np.abs(self.alpha.imag) < self._skiptol)[0]
        if indices.size > 0:
            print('INFO    > Skip some `alpha` imaginary part '
                  + ' (tol={}). '.format(self._skiptol)
                  + 'Max(Im alpha) = {:e}.'.format(np.abs(self.alpha[indices].imag).max()))
        self.alpha[indices] = self.alpha[indices].real + 0j

        if alpha.size < self.Nmodes:
            ValueError('The number of found root is too small.')

        # Instanciate Product object to compute overlap integrals
        self.p = Product(mu, nu)
        self._matching_method = None
        self._sigma = None

    def __repr__(self):
        """Define the representation of the class."""
        r = "Instance of {} with mu={} and nu={}.\n".format(self.__class__.__name__,
                                                            self.mu, self.nu) \
            + "At k={}.".format(self.k)
        return r

    def _K(self, alpha):
        """Dispersion equation expressed in alpha."""
        mu, nu = self.mu, self.nu
        return (mu+nu)*cos(alpha) + sin(alpha)*(alpha - mu*nu/alpha)

    def _Kp(self, alpha, deriv_in_s=True):
        """Derivative with respect to s of the dispersion equation.

        Parameters
        ----------
        alpha: complex
            The transverse wavenumber used in the evaluation.
        deriv_in_s: bool, optional
            If `True` the derivative is done in `alpha`. Default `True`.

        Example
        --------
        >>> guide = Guide(mu=3.08753629148967+3.62341792246081j, nu=3.17816250726595+4.67518038763374j)
        >>> alpha3 = guide.alpha[np.round(guide.multiplicities) == 3][0]
        >>> abs(guide._Kp(alpha3)) < 1e-10
        True

        Remarks
        -------
        The expression is true only if `_K(alpha)` is zero.
        """
        mu, nu = self.mu, self.nu
        s = np.sqrt(self.k**2 - alpha**2, dtype=complex)
        p = mu*nu
        gp = cos(alpha)/alpha - sin(alpha)/alpha**2
        Kp = ((1-nu-mu)*sin(alpha) + alpha*cos(alpha) - gp*p)
        if deriv_in_s:
            Kp *= (-s/alpha)
        return Kp

    def _Kpp(self, alpha):
        """Second order derivative with respect to s of the dispersion equation.

        Parameters
        ----------
        alpha: complex
            The transverse wavenumber used in the evaluation.

        >>> guide = Guide(mu=3.08753629148967+3.62341792246081j, nu=3.17816250726595+4.67518038763374j)
        >>> alpha3 = guide.alpha[np.round(guide.multiplicities) == 3][0]
        >>> abs(guide._Kpp(alpha3)) < 1e-8
        True

        Remarks
        -------
        The expression is true only if `_Kp(alpha)` is zero.
        """
        mu, nu = self.mu, self.nu
        s = np.sqrt(self.k**2 - alpha**2, dtype=complex)
        p = mu*nu
        h = (alpha*cos(alpha) - sin(alpha)) / alpha**3
        # Formula from Emmanuel paper
        Kpp = - self._K(alpha) + 2*cos(alpha) + 2*h*p
        return Kpp*(-s/alpha)**2

    def _Kppp(self, alpha):
        """Third order derivative with respect to s of the dispersion equation.

        Parameters
        ----------
        alpha: complex
            The transverse wavenumber used in the evaluation.

        Remarks
        -------
        The expression is true only if _Kpp(alpha) is zero.
        """
        mu, nu = self.mu, self.nu
        s = np.sqrt(self.k**2 - alpha**2, dtype=complex)
        p = mu*nu
        hp = (-alpha**2*sin(alpha) - 3*alpha*cos(alpha)
              + 3*sin(alpha)) / alpha**4
        # *(-alpha/s) because we want the derivative % alpha
        Kppp = - self._Kp(alpha)*(-alpha/s) - 2*sin(alpha) + 2*hp*p
        return Kppp*(-s/alpha)**3

    def _Y(self, y, n):
        """Compute transverse field.

        Parameters
        ----------
        y : array
            The position where the function is evaluate.
        n : int
            The modal index.
        """
        alphan = self.alpha[n]
        Yn = cos(alphan * y) - self.mu*sin(alphan * y)/alphan
        return Yn

    def _Chi(self, y, n):
        """Compute the additional EP2 wavefunction.

        Parameters
        ----------
        y : array
            The position where the function is evaluate.
        n : int
            The EP3 modal index.
        """
        alphan = self.alpha[n]
        dYn = - alphan * sin(alphan * y) - self.mu*cos(alphan * y)
        Chin = (y - self.mu/(alphan**2+self.mu**2)) * dYn
        return Chin

    def _Chi_check(self, n):
        """Check the BC for the additional function Chi.

        Parameters
        ----------
        n : int
            The EP3 modal index.

        Remarks
        -------
        Should be 0.

        Examples
        --------
        >>> guide = Guide(mu=0., nu=1.650611293539765+2.059981457179885j)
        >>> n = np.where(np.round(guide.multiplicities) == 2)[0].item()
        >>> abs(guide._Chi_check(n)) < 1e-12
        True
        """
        y = 1.
        alphan = self.alpha[n]
        dYn = - alphan * sin(alphan * y) - self.mu*cos(alphan * y)
        d2Yn = - alphan**2 * cos(alphan * y) + self.mu*alphan*sin(alphan * y)
        dyChin = (y - self.mu/(alphan**2+self.mu**2)) * d2Yn + dYn
        return dyChin - self.nu*self._Chi(y, n)

    def _xi(self, y, n):
        """Compute the additional EP3 wavefunction.

        Parameters
        ----------
        y : array
            The position where the function is evaluate.
        n : int
            n should be the EP3 index.
        """
        alphan = self.alpha[n]
        mu = self.mu
        dYn = - alphan * sin(alphan * y) - mu*cos(alphan * y)
        D = alphan**2 + mu**2
        xin = (y - mu/D)**2 * self._Y(y, n) - 2*mu/D**2 * dYn
        return xin

    def _xi_check(self, n):
        """Check the BC for the additional function xi.

        Parameters
        ----------
        n : int
            The EP3 modal index.

        Remarks
        -------
        Should be 0.

        Examples
        --------
        >>> guide = Guide(mu=3.08753629148967+3.62341792246081j, nu=3.17816250726595+4.67518038763374j)
        >>> n = np.where(np.round(guide.multiplicities) == 3)[0].item()
        >>> abs(guide._xi_check(n)) < 5e-10
        True
        """
        y = 1.
        alphan = self.alpha[n]
        D = alphan**2 + self.mu**2
        dYn = - alphan * sin(alphan * y) - self.mu*cos(alphan * y)
        d2Yn = - alphan**2 * cos(alphan * y) + self.mu*alphan*sin(alphan * y)
        dyxin = (2*(y - self.mu/D) * self._Y(y, n)
                 + dYn * (y - self.mu/D)**2
                 - 2*self.mu/D**2 * d2Yn)
        return dyxin - self.nu*self._xi(y, n)

    def P(self, n, method='ana'):
        """Compute the modal norm.

        Parameters
        ----------
        n : int
            The modal index.
        method : string
            Choose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> guide = Guide(1., 2j)
        >>> abs(guide.P(0, method='ana') - guide.P(0, method='num')) < 1e-10
        True
        """
        if method == 'ana':
            alphan = self.alpha[n]
            sn = np.sqrt(self.k**2 - alphan**2, dtype=complex)
            P = - self._Y(1, n) / (2*sn) * self._Kp(alphan)
        elif method == 'num':
            y = self.p.y
            Yn = self._Y(y, n)
            P = (Yn * Yn) @ self.p.w
        else:
            raise ValueError('`method` argument is not recognized')
        return P

    def Q(self, n, method='ana'):
        """Compute modal norm at EP2.

        Parameters
        ----------
        n : int
            The modal index.
        method : string
            Choose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> guide = Guide(mu=0., nu=1.650611293539765+2.059981457179885j)
        >>> n = np.where(np.round(guide.multiplicities) == 2)[0].item()
        >>> abs(guide.Q(n, method='ana') - guide.Q(n, method='num')) < 1e-10
        True

        """
        if method == 'ana':
            alphan = self.alpha[n]
            sn = np.sqrt(self.k**2 - alphan**2, dtype=complex)
            Q = alphan**2*self._Y(1, n) / (4*sn**2) * self._Kpp(alphan)
        elif method == 'num':
            y = self.p.y
            Yn = self._Y(y, n)
            # Q ~ 0 if n is an EP. Be carefull with the present implementation
            # since n is the same for both function !
            Q = (self._Chi(y, n) * Yn) @ self.p.w
        else:
            raise ValueError('`method` argument is not recognized')
        return Q

    def R(self, n, m=None, method='ana'):
        """Compute modal norm at an EP3.

        Parameters
        ----------
        n : int
            Index of the triple roots.
        m : int
            Extra index to check orthogonality relation for validation purpose.
            Default `None`.
        method : string
            Choose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> guide = Guide(mu=3.08753629148967+3.62341792246081j, nu=3.17816250726595+4.67518038763374j)
        >>> n = np.where(np.round(guide.multiplicities) == 3)[0].item()
        >>> abs(guide.R(n, method='ana') - guide.R(n, method='num')) < 1e-10
        True

        """
        if method == 'ana':
            alphan = self.alpha[n]
            sn = np.sqrt(self.k**2 - alphan**2, dtype=complex)
            R = alphan**2/(6*sn**3) * self._Y(1, n) * self._Kppp(alphan)
        elif method == 'num':
            if m is None:
                m = n
            y = self.p.y
            Ym = self._Y(y, m)
            R = (self._xi(y, n) * Ym) @ self.p.w
        else:
            raise ValueError('`method` argument is not recognized')
        return R

    def _T(self, n, method='num'):
        r"""Compute \(\int_0^1 \chi(y)^2 dy \).

        Parameters
        ----------
        n : int
            Index of the triple roots.
        method : string
            Choose a method to compute the integral. Should be in {'num'}.

        Examples
        --------
        No example yet.
        """
        if method == 'num':
            y = self.p.y
            T = (self._Chi(y, n) * self._Chi(y, n)) @ self.p.w
        else:
            raise ValueError('`method` argument is not recognized.')
        return T

    def _U(self, n, method='num'):
        r"""Compute of \(\int_0^1 \xi(y)\chi(y) dy \).

        Parameters
        ----------
        n : int
            Index of the triple roots.
        method : string
            Choose a method to compute the integral. Should be in {'num'}.

        Examples
        --------
        No example yet.

        """
        if method == 'num':
            y = self.p.y
            U = (self._xi(y, n) * self._Chi(y, n)) @ self.p.w
        else:
            raise ValueError('`method` argument is not recognized.')
        return U

    def _W(self, n, method='num'):
        r"""Compute \(\int_0^1 \xi(y)^2 dy \).

        Parameters
        ----------
        n : int
            Index of the triple roots.
        method : string
            Choose a method to compute the integral. Should be in {'num'}.

        Examples
        --------
        No example yet.

        """
        if method == 'num':
            y = self.p.y
            W = (self._xi(y, n)**2) @ self.p.w
        else:
            raise ValueError('`method` argument is not recognized.')
        return W

    def _Ups(self, n):
        r"""Compute \(\Upsilon(s)=\frac{2s^2}{\alpha^2(\alpha^2+\mu^2)}-\frac{\alpha^2+3s^2}{\alpha^4}\).

        Parameters
        ----------
        n : int
            The index of the triple mode.
        """
        alphan = self.alpha[n]
        sn = np.sqrt(self.k**2 - alphan**2, dtype=complex)
        mu = self.mu
        D = (alphan**2 + mu**2)
        Ups = (2*sn**2) / (alphan**2 * D) - (alphan**2 + 3*sn**2)/alphan**4
        return Ups

    def solve_disp(self, R=10, Npz=300, Ni=50000, method='hybrid', display=False):
        """Use contour integral to solve the dispersion equation.

        Parameters
        ----------
        R : float
            The maximal radius of the contour.
        Npz : int
            An upper bound for the number of roots
        Ni : int
            The number of of integration points in the countour.
        method : string, optional
            Define the method to solve the disperison equation. `'contour'`
            implies that only contour integrals are used; `'hybrid'` implies
            that contour integrals are used to find roots inside the C(0,R)
            and if the number of roots is  smaller than `Nmodes`, assymptotic
            expressions are used to get all the requested modes.

        Examples
        --------
        Check on triple roots case.
        >>> guide = Guide(3.1781625072659+4.67518038763374j, 3.08753629148967+3.62341792246081j)
        >>> np.abs(guide.alpha - (4.19693888263064 - 2.60864153789443j) ).min() < 1e-10
        True
        """
        refine = False
        tol_check = 1e-5

        def dK(alpha):
            """Compute the derivative of K with respect to alpha."""
            return self._Kp(alpha, deriv_in_s=False)

        # solve with moment method
        # polze.logger.setLevel(20)
        # pz = polze.PZ((self._K, dK), Rmax=R, Npz=Npz, Ni=Ni, split=True,
        pz = polze.PZ(self._K, Rmax=R, Npz=Npz, Ni=Ni, split=True,
                      options={'_vectorized': True, '_Npz_limit': 8,
                               '_zeros_only': True,
                               '_tol': 1e-4})
        self._pz = pz
        pz.solve()
        if display:
            pz.display()
        _, (alphan, multiplicities) = pz.dispatch(refine=refine, multiplicities=True)
        status = pz.check(tol=tol_check)
        if not status.all_roots:
            print('Warning some root are inacurate w.r.t. tol={}'.format(tol_check))
        # all modes come by pair...
        pos = alphan.real > self._alpha_pos_tol
        alphan = alphan[pos]
        multiplicities = multiplicities[pos]
        # return sorted in real multiplicities ascending order
        ind = np.argsort(alphan.real, axis=0)
        self.alpha = np.take_along_axis(alphan, ind, axis=0)
        self.multiplicities = np.take_along_axis(multiplicities, ind, axis=0)

        if method == 'hybrid':
            # Use assymptotic if the number of found roots is too small
            if self.alpha.size < self.Nmodes:
                n = np.arange(np.round(self.alpha[-1]/np.pi).real.astype(int) + 1,
                              self.Nmodes + 2)
                alpha_as = self.alpha_asymptotics(n, refine=True)
                mult_as = np.ones_like(alpha_as)
                self.alpha = np.concatenate((self.alpha, alpha_as))
                self.multiplicities = np.concatenate((self.multiplicities, mult_as))

        return self.alpha.copy()

    def alpha_asymptotics(self, n, refine=False):
        """Compute transverse wavenumbers using asymptotics.

        Parameters
        ----------
        n : array
            The request modal index.
        refine : bool, optional
            Use Newton method to refine asymptotic values.

        Return
        ------
        alphan : array
            The value of the wavenumber.
        """
        same_tol = 1.
        p = self.mu * self.nu
        s = self.mu + self.nu
        npi = n*np.pi
        alphan = npi - s/npi + (s**3/3-s**2-p*s)/(npi**3)
        # provide alpha derivative of K
        dK = lambda z: self._Kp(z, deriv_in_s=False)
        if refine:
            # Use stand root finding (Newton) to improve it
            for i, alpha in enumerate(alphan):
                out = spop.root_scalar(f=self._K, method='newton',
                                       x0=alpha, fprime=dK)
                if out.converged:
                    alphan[i] = out.root
                    if abs(out.root - alpha) > same_tol:
                        print(' > NR converged to another alpha for n={}.'.format(i))
                else:
                    print(' > Warning NR has not converged for n={}.'.format(i))
        return alphan

    @staticmethod
    def eps(n):
        """Return 2 if n==0 and 1 otherwise."""
        if n == 0:
            return 2
        else:
            return 1

    def get_axial_wn_lined(self):
        """Return the axial wavenumber in the lined domain."""
        alphan = self.alpha
        return np.sqrt(self.k**2 - alphan**2, dtype=complex)

    def get_index_at_multiplicity(self, n, first=True):
        """Return the index of the mode with the `n` multiplicty.

        Parameters
        ----------
        n : int
            Multiplicity value.
        first : bool, optional
            If `first=True` only the first found index value is return, else
            the whole vector is returned. The default is True.

        Returns
        -------
        index : int or array
            Index of the modes satisfaying the condition.
        """
        if first:
            # The [0] keep just the first dimension
            index = np.where(np.round(self.multiplicities) == n)[0][0]
        else:
            index = np.where(np.round(self.multiplicities) == n)[0]

        return index

    def matching(self, EP_order):
        """Compute the reflected and the transmitted waves for the EP_order.

        This function is a common interface to all `matching_*` method.

        Parameters
        ----------
        EP_order : int
            The order of the EP. It should be between 0 (no EP) and 3.
        """

        if EP_order == 3:
            self.matching_EP3()
        elif EP_order == 2:
            self.matching_EP2()
        elif EP_order == 0:
            mult = np.round(self.multiplicities.real)
            if (mult > 1).any():
                print('WARNING > Some roots have multiplicties > 1. Try an other matching strategy.')
            self.matching_std()
        else:
            raise ValueError('`EP_order` must be between 0 and 3.')

    def matching_std(self):
        """Compute the reflected and the transmitted waves for simple roots.

        The incident field is a plane wave.
        A and B denotes the refected and the transmitted waves respectivelly.
        """
        self._matching_method = 'std'
        Nmodes = self.Nmodes
        rhs = np.zeros((2*Nmodes,), dtype=complex)
        # Eq. (16)
        I = np.eye(Nmodes)
        MB = np.zeros((Nmodes, Nmodes), dtype=complex)
        for (m, n), _ in np.ndenumerate(MB):
            MB[m, n] = -(2./self.eps(m)) * self.L(m, n)
        rhs[0] = -1.

        # Eq. (18)
        MA = np.zeros((Nmodes, Nmodes), dtype=complex)
        for j in range(0, Nmodes):
            alphaj = self.alpha[j]
            sj = np.sqrt(self.k**2 - alphaj**2, dtype=complex)
            Pj = self.P(j)
            rhs[Nmodes+j] = self.k*self.L(0, j) / (sj*Pj)
            for n in range(0, Nmodes):
                etan = np.sqrt(self.k**2 - (n*np.pi)**2, dtype=complex)
                MA[j, n] = 1./(sj*Pj) * etan * self.L(n, j)

        # Build global system
        K = np.block([[I, MB],
                      [MA, I]])
        sol = np.linalg.solve(K, rhs)
        A = sol[0:Nmodes]
        B = sol[Nmodes::]
        self.A, self.B = A, B
        # Store matrix for latter inspection
        self._matching_matrix = K.copy()
        return A, B

    def matching_EP2(self):
        """Compute the reflected and the transmitted waves amplitude for EP2.

        The incident field is a plane wave.
        A and B denotes the refected and the transmitted waves respectivelly.
        The amplitude of the additionnal wavefunction is put at the end of B
        vector.

        Remarks
        -------
        Assume there is only one EP2.
        """
        # Check if there is an EP2
        if ((np.max(np.round(self.multiplicities)) > 1)
           and (np.max(np.round(self.multiplicities)) < 3)):
            # Find EP2 index (assume there is only one)
            nEP2 = np.where(np.round(self.multiplicities) == 2)[0].item()
            self._matching_method = 'EP2'
        else:
            raise ValueError('No root found with `multiplicities == 2`')
        # define wavenumver at the EP2
        alphab = self.alpha[nEP2]
        sb = np.sqrt(self.k**2 - alphab**2, dtype=complex)
        Nmodes = self.Nmodes
        # Dimensions are modified du to EP2
        rhs = np.zeros((2*Nmodes-1,), dtype=complex)
        # Eq. (24)
        I = np.eye(Nmodes)
        MB = np.zeros((Nmodes, Nmodes+1), dtype=complex)
        for m in range(0, Nmodes):
            # standard contributions
            for n in range(0, Nmodes):
                MB[m, n] = -(2./self.eps(m)) * self.L(m, n)
            # additional wavefunction contribution
            MB[m, Nmodes] = - 2*sb/(self.eps(m)*alphab**2)*self.M(m, nEP2)
        rhs[0] = -1.

        # Eq. (24)
        IB = np.zeros((Nmodes-1, Nmodes+1), dtype=complex)
        MA = np.zeros((Nmodes-1, Nmodes), dtype=complex)
        index = [j for j in range(0, Nmodes) if j != nEP2]
        for line, j in enumerate(index):
            alphaj = self.alpha[j]
            sj = np.sqrt(self.k**2 - alphaj**2, dtype=complex)
            Pj = self.P(j)
            rhs[Nmodes+line] = self.k*self.L(0, j) / (sj*Pj)
            IB[line, j] = 1.
            for n in range(0, Nmodes):
                etan = np.sqrt(self.k**2 - (n*np.pi)**2, dtype=complex)
                MA[line, n] = 1./(sj*Pj) * etan * self.L(n, j)
        # Define the 2 additional equations Eq. (27)
        MB2 = np.zeros((2, Nmodes + 1), dtype=complex)
        MA2 = np.zeros((2, Nmodes), dtype=complex)
        MB2[0, nEP2] = sb*self.Q(nEP2)
        MB2[0, Nmodes] = (sb/alphab)**2*self._T(nEP2) - self.Q(nEP2)
        MB2[1, Nmodes] = (sb/alphab)**2*self.Q(nEP2)
        for n in range(0, Nmodes):
            etan = np.sqrt(self.k**2 - (n*np.pi)**2, dtype=complex)
            MA2[0, n] = etan * self.M(n, nEP2)
            MA2[1, n] = etan * self.L(n, nEP2)
        # Addition terms in the RHS
        rhs2 = np.array([self.k*self.M(0, nEP2),
                         self.k*self.L(0, nEP2)])
        # Build global system
        K = np.block([[I, MB],
                      [MA, IB],
                      [MA2, MB2]])
        RHS = np.block([rhs, rhs2])
        sol = np.linalg.solve(K, RHS)
        A = sol[0:Nmodes]
        B = sol[Nmodes::]
        self.A, self.B = A, B
        # Store matrix for latter inspection
        self._matching_matrix = K.copy()
        return A, B

    def matching_EP3(self):
        """Compute the reflected and the transmitted waves amplitude for EP3 case.

        The incident field is a plane wave.
        A and B denotes the refected and the transmitted waves respectivelly.
        The amplitude of the additionnal wavefunctions are put at the end of B
        vector.
        """
        # Check if there is an EP3
        if np.max(np.round(self.multiplicities)) == 3:
            # Find EP2 index (assume there is only one)
            nEP3 = np.where(np.round(self.multiplicities) == 3)[0].item()
            self._matching_method = 'EP3'
        else:
            raise ValueError('No root found with `multiplicities == 3`')
        # define wavenumver at the EP3
        alphab = self.alpha[nEP3]
        sb = np.sqrt(self.k**2 - alphab**2, dtype=complex)
        Nmodes = self.Nmodes
        mu = self.mu
        # Dimensions are modified du to EP3
        rhs = np.zeros((2*Nmodes-1,), dtype=complex)
        # Eq. (34)
        I = np.eye(Nmodes)
        MB = np.zeros((Nmodes, Nmodes+2), dtype=complex)
        for m in range(0, Nmodes):
            # standard contributions
            for n in range(0, Nmodes):
                MB[m, n] = -(2./self.eps(m)) * self.L(m, n)
            # additional wavefunction contribution
            MB[m, Nmodes] = - 2*sb/(self.eps(m)*alphab**2)*self.M(m, nEP3)
            MB[m, Nmodes+1] = -(2/self.eps(m))*(- sb**2/alphab**2*self.N(m, nEP3)
                                                + self._Ups(nEP3)*self.M(m, nEP3))
        rhs[0] = -1.

        # Eq. (36)
        IB = np.zeros((Nmodes-1, Nmodes+2), dtype=complex)
        MA = np.zeros((Nmodes-1, Nmodes), dtype=complex)
        index = [j for j in range(0, Nmodes) if j != nEP3]
        for line, j in enumerate(index):
            alphaj = self.alpha[j]
            sj = np.sqrt(self.k**2 - alphaj**2, dtype=complex)
            Pj = self.P(j)
            rhs[Nmodes+line] = self.k*self.L(0, j) / (sj*Pj)
            IB[line, j] = 1.
            for n in range(0, Nmodes):
                etan = np.sqrt(self.k**2 - (n*np.pi)**2, dtype=complex)
                MA[line, n] = 1./(sj*Pj) * etan * self.L(n, j)
        # Define the 3 additional equations Eq. (37)
        MB2 = np.zeros((3, Nmodes + 2), dtype=complex)
        MA2 = np.zeros((3, Nmodes), dtype=complex)
        R = self.R(nEP3)
        T = self._T(nEP3)
        U = self._U(nEP3)
        W = self._W(nEP3)
        Ups = self._Ups(nEP3)
        D = alphab**2 + mu**2
        Theta = (sb*Ups*U - 2*(sb/alphab**2)*U
                 - (sb**3/alphab**2)*W
                 + 2*mu**2*sb*R/(alphab**2*D))
        Omega = (sb*Ups*T - 2*sb/alphab**2*T
                 - (sb**3/alphab**2)*U)
        # line 1
        MB2[0, nEP3] = sb*R
        MB2[0, Nmodes] = (sb**2/alphab**2)*U - R
        MB2[0, Nmodes+1] = Theta
        # line 2
        MB2[1, Nmodes] = (sb**2/alphab**2)*T
        MB2[1, Nmodes+1] = Omega
        # line 3
        MB2[2, Nmodes+1] = -(sb**3/alphab**2)*R
        for n in range(0, Nmodes):
            etan = np.sqrt(self.k**2 - (n*np.pi)**2, dtype=complex)
            MA2[0, n] = etan * self.N(n, nEP3)
            MA2[1, n] = etan * self.M(n, nEP3)
            MA2[2, n] = etan * self.L(n, nEP3)
        # Addition terms in the RHS
        rhs2 = np.array([self.k*self.N(0, nEP3),
                         self.k*self.M(0, nEP3),
                         self.k*self.L(0, nEP3)])
        # Build global system
        K = np.block([[I, MB],
                      [MA, IB],
                      [MA2, MB2]])
        RHS = np.block([rhs, rhs2])
        sol = np.linalg.solve(K, RHS)
        A = sol[0:Nmodes]
        B = sol[Nmodes::]
        self.A, self.B = A, B
        # Store matrix for latter inspection
        self._matching_matrix = K.copy()
        return A, B

    def _build_field(self, x1, y1, x2, y2, F=1.):
        """Build physical field from modal expansion.
        """
        # Define a map between matching strategy and the potential Psi_2
        multiplicty = {'std': 1, 'EP2': 2, 'EP3': 3}
        # Init field
        Psi1 = F*np.exp(1j*self.k*x1)
        Psi2 = np.zeros_like(x2, dtype=complex)
        v1 = F*np.exp(1j*self.k*x1) * 1j * self.k
        v2 = np.zeros_like(x2, dtype=complex)
        for n in range(0, self.Nmodes):
            sn = np.sqrt(self.k**2 - self.alpha[n]**2, dtype=complex)
            etan = np.sqrt(self.k**2 - (n*np.pi)**2, dtype=complex)
            # reconstruct Psi (potential)
            temp1 = self.A[n] * np.exp(-1j*etan*x1) * cos(n * np.pi * y1)
            # temp2 = self.B[n] * np.exp(1j*sn*x2) * self._Y(y2, n)
            temp2 = self._psi2n(n, x2.T, y2.T)[0].T
            Psi1 += temp1
            Psi2 += temp2
            # reconstruct the velicity field
            v1 += -temp1 * 1j * etan
            v2 += temp2 * 1j * sn

        # Add additional waveform contribution for EPs
        # if EPx>=2
        if multiplicty[self._matching_method] >= 2:
            nEP2 = np.where(np.round(self.multiplicities) >= 2)[0].item()
            # meshgrid have not the same orientation
            wf2, dx_wf2 = self._add_wf2(nEP2, x2.T, y2.T)
            Psi2 += wf2.T
            v2 += dx_wf2.T

        if multiplicty[self._matching_method] >= 3:
            nEP3 = np.where(np.round(self.multiplicities) >= 3)[0].item()
            wf3, dx_wf3 = self._add_wf3(nEP3, x2.T, y2.T)
            Psi2 += wf3.T
            v2 += dx_wf3.T

        return Psi1, v1, Psi2, v2

    def plot_intensity(self, Ny=50, Nx=50, F=1., xmax=5):
        """Plot intensity field for x>0 from modal expansion."""
        y = np.linspace(0., 1., Ny)
        x2, y2 = np.meshgrid(np.linspace(0, xmax, Nx), y)
        intensity = np.zeros_like(x2)
        for i in range(0, Nx):
            intensity[:, i] = self.transmitted_power(x2[0, i], intensity_only=True, y=y)
        # plot
        fig = plt.figure('Intensity field in domain 2')
        ax = plt.gca()
        lined = ax.pcolormesh(x2, y2, intensity, shading='auto')
        fig.colorbar(lined)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.axis('scaled')
        return intensity

    def get_L2_error(self, NGP=100):
        """Compute the L2 Error at the interface.

        Parameters
        ----------
        NGP : int, optional
            Number of Gauss Point used in the integration. The default is 100.

        Returns
        -------
        Err2_Psi : float
            The value of the normalized L2 Error for the potential.
        Err2_v : float
            The value of the normalized L2 Error for the axial velocity.
        """
        y, w = Product._gauleg(NGP, a=0, b=1)
        x, y = np.meshgrid(0., y)
        Psi1, v1, Psi2, v2 = self._build_field(x, y, x, y)
        # Potential
        ref_Psi1 = np.sqrt(w @ np.abs(Psi1)**2)
        Err2_Psi = np.sqrt(w @ np.abs(Psi1 - Psi2)**2) / ref_Psi1
        # axial velocity
        ref_v1 = np.sqrt(w @ np.abs(v1)**2)
        Err2_v = np.sqrt(w @ np.abs(v1 - v2)**2) / ref_v1
        return Err2_Psi, Err2_v

    def get_cond2(self):
        """Compute the condition number of the matching matrix."""
        try:
            return np.linalg.cond(self._matching_matrix, 2)
        except AttributeError:
            print('Call a `matching_*` method before.')

    def filter_coefs(self, filter_name='Lanczos'):
        """Filter the modal coefficient in both domain to improve convergence.

        The filtering is performed in place, thus all acces to modal amplitude
        in post-process will be affected.

        Parameters
        ----------
        filter_name : string, optional
            Name of a valid filter ('lanczos', ...). If 'undo' is used, the
            initial coefficient are recovered. If 'show' is used, the active
            filter is return. If 'list'' is used all available filters are
            return. The default is 'Lanczos'.
        """
        if filter_name in CreateFilter.keys():
            # If filter has not yet been applied, we need to store raw values
            if not (hasattr(self, '_A_raw') and hasattr(self, '_B_raw')):
                self._A_raw = self.A.copy()
                self._B_raw = self.B.copy()
            # Apply new filer
            self._sigma = CreateFilter[filter_name](self.Nmodes)
            sigma = self._sigma.filter()
            self.A[:self.Nmodes] = sigma * self._A_raw[:self.Nmodes]
            self.B[:self.Nmodes] = sigma * self._B_raw[:self.Nmodes]

        elif (filter_name == 'undo') and (self._sigma is not None):
            self._sigma = None
            self.A = self._A_raw.copy()
            self.B = self._B_raw.copy()

        elif filter_name == 'show':
            if self._sigma is not None:
                print("'%s' filter is activate." % self._sigma.name())

        elif filter_name == 'list':
            print(CreateFilter.keys())



        else:
            print("'filter_name' value are not supported.")

    def plot_field(self, only_interface=True, Ny=50, Nx=50, markerevery=5,
                   inset=False, F=1):
        """Plot potential field to check the continuity.

        Parameters
        ----------
        only_interface: bool, optional
             Plot only the fields on the matching interface. Default `True`.
        Nx : int, optional
             Number of grid point in the x direction.  Default 50.
        Ny : int, optional
             Number of grid point in the y direction. Default 50.
        markerevery : int, optional
            Number of point to skip between two markers. Default 5.
        inset : bool
            If `True`, add a zoom on the field at y=0. Parameters are
            mostly manualy tuned below.
        F : complex, optional
            Amplitude of the incident plane wave. Not supposed to be changed yet.
            Default 1.
        """
        lw = 1.  # linewidth
        ms = 4.  # markersize
        ylim_zoom = 0.05  # y upper bound of the zoom
        ylim_v_zoom = ylim_zoom/2.5
        # Create plotting mesh
        if only_interface:
            Nx = 2
        y = np.linspace(0., 1., Ny)
        x1, y1 = np.meshgrid(np.linspace(-5, 0, Nx), y)
        x2, y2 = np.meshgrid(np.linspace(0, 5, Nx), y)
        # Build field
        Psi1, v1, Psi2, v2 = self._build_field(x1, y1, x2, y2, F=F)
        # Plotting
        if not only_interface:
            fig = plt.figure('Psi field')
            ax = plt.gca()
            clim = [0,
                    max(np.abs(Psi1).max(), np.abs(Psi2).max())]
            air = ax.pcolormesh(x1, y1, np.abs(Psi1), shading='auto',
                                vmin=clim[0], vmax=clim[1])
            ax.pcolormesh(x2, y2, np.abs(Psi2), shading='auto',
                          vmin=clim[0], vmax=clim[1])
            # fig.colorbar(air)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.axis('scaled')

            fig1d_psi = plt.figure('Psi in the midle of the duct')
            plt.plot(x1[Nx//2, :], Psi1[Nx//2, :].real, 'k-')
            # plt.plot(x1[Nx//2, :], np.abs(Psi1[Nx//2, :]), 'k-.')
            plt.plot(x1[Nx//2, :], Psi1[Nx//2, :].imag, 'k:')

            plt.plot(x2[Nx//2, :], Psi2[Nx//2, :].real, 'b-')
            # plt.plot(x2[Nx//2, :], np.abs(Psi2[Nx//2, :]), 'b-.')
            plt.plot(x2[Nx//2, :], Psi2[Nx//2, :].imag, 'b:')

            plt.xlabel('x (m)')
            plt.ylabel(r'$\Psi$')

            fig1d_v = plt.figure('v in the midle of the duct')
            plt.plot(x1[Nx//2, :], v1[Nx//2, :].real, 'k-')
            plt.plot(x1[Nx//2, :], v1[Nx//2, :].imag, 'k:')

            plt.plot(x2[Nx//2, :], v2[Nx//2, :].real, 'b-')
            plt.plot(x2[Nx//2, :], v2[Nx//2, :].imag, 'b:')
            plt.xlabel('x (m)')
            plt.ylabel(r'$v_x$')

        # Potential
        fig1dy_psi = plt.figure(r'$\psi$ at the interface')
        ax_psi = fig1dy_psi.add_subplot(111)
        if inset:
            axins_psi = zoomed_inset_axes(ax_psi, 6, loc=6, borderpad=5)
            axes_list = [ax_psi, axins_psi]
        else:
            axes_list = [ax_psi]
        # loop to plot on the axes and its inset
        for ax in axes_list:
            ax.plot(y, Psi1[:, -1].real, 'b-', linewidth=lw, label=r'Re ${\psi_1}$')
            ax.plot(y, Psi1[:, -1].imag, 'r-', linewidth=lw, label=r'Im ${\psi_1}$')
            ax.plot(y, Psi2[:, 0].real, 'b:', linewidth=lw, marker='.', markersize=ms,
                    markevery=markerevery, label=r'Re ${\psi_2}$')
            ax.plot(y, Psi2[:, 0].imag, 'r:', marker='.', linewidth=lw,
                    markevery=markerevery, markersize=ms, label=r'Im ${\psi_2}$')
            ax.plot(y, np.abs(Psi2[:, 0]), 'k:', marker='.', linewidth=lw,
                    markevery=markerevery, markersize=ms, label=r'$|{\psi_2}|$')
        if inset:
            mark_inset(ax_psi, axins_psi, loc1=1, loc2=3, fc="none", ec="0.5")
            axins_psi.set_xlim(0, ylim_zoom)  # Limit the region for zoom
            psi_zoom_max = Psi1[:, -1][y > ylim_zoom][0].real
            psi_zoom_min = Psi1[0, -1].real
            axins_psi.set_ylim(sorted([psi_zoom_min, psi_zoom_max]))
            plt.xticks(visible=False)  # Not present ticks
            plt.yticks(visible=False)
        ax_psi.set_xlabel('$y$')
        ax_psi.set_ylabel(r'$\psi$')
        ax_psi.legend()

        # Velocity
        fig1dy_dpsi = plt.figure(r'$\partial_x\psi$ at the interface')
        ax_dpsi = fig1dy_dpsi.add_subplot(111)
        if inset:
            axins_dpsi = zoomed_inset_axes(ax_dpsi, 3, loc='upper left',
                                           bbox_to_anchor=(0.15, 0.95),
                                           bbox_transform=ax_dpsi.transAxes, borderpad=0)
            axes_list = [ax_dpsi, axins_dpsi]
        else:
            axes_list = [ax_dpsi]
        # loop to plot on the axes and its inset
        for ax in axes_list:
            ax.plot(y, v1[:, -1].real, 'b-', linewidth=lw, label=r'Re ${\partial_x\psi_1}$')
            ax.plot(y, v1[:, -1].imag, 'r-', linewidth=lw, label=r'Im ${\partial_x\psi_1}$')
            ax.plot(y, v2[:, 0].real, 'b:', linewidth=lw, marker='.', markevery=markerevery,
                    markersize=ms, label=r'Re ${\partial_x\psi_2}$')
            ax.plot(y, v2[:, 0].imag, 'r:', linewidth=lw, marker='.', markevery=markerevery,
                    markersize=ms, label=r'Im ${\partial_x\psi_2}$')
        if inset:
            mark_inset(ax_dpsi, axins_dpsi, loc1=2, loc2=3, fc="none", ec="0.5")
            axins_dpsi.set_xlim(-0.5e-2, ylim_v_zoom)  # Limit the region for zoom
            v_zoom_max = v2[:, 0][y > ylim_v_zoom][0].imag
            v_zoom_min = v2[0, 0].imag
            bounds_v = sorted([v_zoom_min, v_zoom_max])
            axins_dpsi.set_ylim([bounds_v[0]*0.98, bounds_v[1]*1.01])
            plt.xticks(visible=False)  # Not present ticks
            plt.yticks(visible=False)
        ax_dpsi.set_xlabel('$y$')
        ax_dpsi.set_ylabel(r'$\partial_x\psi$')
        ax_dpsi.legend()
        return ax_dpsi, ax_psi

    def liner_integral(self, x):
        r"""Compute line integral along the liner that appears in E_2.

        Compute
        A = \( \int_0^x \mathrm{Im} (\mu |\psi_2(x',0)|^2 + \nu |\psi_2(x',1)|^2)  dx'\)
        using Simpson formula (equally space grid, resuse field value...).
        If x is too coarse, results may be inacurate.

        Parameters
        ----------
        x : array
            Abscisses value where the intergral must be return. x[0] must be 0.

        Returns
        -------
        A : array
            Value of the integral for all `x` positions.

        Example
        -------
        >>> guide = Guide(mu=1.0119407382877632+4.602904394703379j, nu=1.0119407382877632-4.602904394703379j, Nmodes=10)
        INFO    >...
        >>> a = guide.matching_EP3()
        >>> E1 = 1 + guide.reflected_power()
        >>> x = np.linspace(0, .2, 50)
        >>> E2 = guide.transmitted_power(x)
        >>> Aref = E1 - E2
        >>> A = guide.liner_integral(x)
        >>> np.linalg.norm(A - Aref) < 1e-10
        True
        """
        def eval_psi(x):
            X, Y = np.meshgrid(x, [0., 1.])
            _, _, Psi2, _ = self._build_field(x1=0, y1=0,
                                              x2=X, y2=Y)
            psi0 = Psi2[0, :]
            psi1 = Psi2[1, :]
            y = (self.mu*np.abs(psi0)**2 +
                 self.nu*np.abs(psi1)**2).imag
            return y

        A = np.zeros_like(x, dtype=float)
        y = eval_psi(x)
        for i in range(1, x.size):
            A[i] = simpson(y[:i], x[:i])
        # Example with adaptative rule
        # Aq = quad(eval_psi, 0, x[-1]
        return A

    def reflected_power(self):
        """Compute the reflected power at the given frequency."""
        F = 1.
        omega = self.k
        # p = 1i * omega * rho * Psi
        # vx = \partial_x Psi
        Winc = 0.5*omega*self.k*abs(F)**2
        W1 = 0.
        for n in range(0, self.Nmodes):
            if n == 0:
                Norm = 1.
            else:
                Norm = 0.5
            etan = np.sqrt(self.k**2 - (n*np.pi)**2, dtype=complex)
            # print(n, ' ', etan)
            # reconstruct Psi (potential)
            W1 += 0.5 * abs(self.A[n])**2 * np.real(1j*omega*np.conj(-1j*etan)) * Norm
        return W1/Winc

    def transmitted_power(self, x, intensity_only=False, y=None):
        """Compute the transmitted power at the given frequency.

        Due to the lack of orthogonality between $Y_n(y)$ and $Y_n^*(y)$,
        the power is best calculated numerically using Gauss Point.

        Parameters
        ----------
        x : array
            Position where the power is computed. `x` should be positive.
        intensity_only : bool [optional]
            If `True` return the intensity instead of the power at the position
            `x` and `y`. `y` should be given. Otherwise, `y` is ignored and
            Gauss point are used in the integation.
        y : array
            y position used to compute the intensity if `intensity_only=True`.
            else, it is set tp the Gauss point value.
        """
        # Check if matching has been done
        if self._matching_method is None:
            raise ValueError('Need to performed the matching before.')
        F = 1.
        omega = self.k
        # Number of Modes involed in the Power computation.
        # can be smaller than Nmodes due to the exponential decay.
        Nmodes = self.Nmodes
        NmodesPower = Nmodes

        # p = 1i * omega * rho * Psi
        # vx = \partial_x Psi
        Winc = 0.5*omega*self.k*abs(F)**2
        W2 = 0.
        # Define a map between matching strategy and the potential Psi_2
        multiplicty = {'std': 1, 'EP2': 2, 'EP3': 3}
        # Get the Gauss points and Gauss weight
        if not intensity_only:  # Compute power
            y = self.p.y
            w = self.p.w
        else:  # Compute intensity at y coordinate
            if y is None:
                raise ValueError('For intensity compution, `y` should be given.')
            # Set the gauss weight to identity to do nothing...
            w = np.eye(y.size)
        # create a mesh. y should varies along the column (Gauss Sum)
        Y, X = np.meshgrid(y, x)
        # In all case \int_0^1 Yn \bar{Ym} dy
        # Bi-orthogonality relation doesn't apply here. Use brute force
        for n in range(0, NmodesPower):
            psin, dx_psin = self._psi2n(n, X, Y)
            for m in range(0, NmodesPower):
                psim, dx_psim = self._psi2n(m, X, Y)
                W2 += 0.5 * omega * np.real(1j * (psin * dx_psim.conj()) @ w)
        # if EPx>=2
        if multiplicty[self._matching_method] >= 2:
            nEP2 = np.where(np.round(self.multiplicities) >= 2)[0].item()
            wf2, dx_wf2 = self._add_wf2(nEP2, X, Y)
            # \bar{B1}\bar{B1}*
            W2 += 0.5 * omega * np.real(1j * (wf2 * dx_wf2.conj()) @ w)
            for n in range(0, NmodesPower):
                psi2n, dx_psi2n = self._psi2n(n, X, Y)
                W2 += 0.5 * omega * np.real(1j * (wf2 * dx_psi2n.conj()) @ w)
                W2 += 0.5 * omega * np.real(1j * (psi2n * dx_wf2.conj()) @ w)
        # if EPx>=3
        if multiplicty[self._matching_method] >= 3:
            nEP3 = np.where(np.round(self.multiplicities) == 3)[0].item()
            # create a mesh. y should varies along the column (Gauss Sum)
            wf3, dx_wf3 = self._add_wf3(nEP3, X, Y)
            # \bbar{B1}\bbar{B1}*
            W2 += 0.5 * omega * np.real(1j * (wf3 * dx_wf3.conj()) @ w)
            # \bar{B1}\bbar{B1}*
            W2 += 0.5 * omega * np.real(1j * (wf2 * dx_wf3.conj()) @ w)
            # \bbar{B1}\bar{B1}*
            W2 += 0.5 * omega * np.real(1j * (wf3 * dx_wf2.conj()) @ w)
            for n in range(0, NmodesPower):
                psi2n, dx_psi2n = self._psi2n(n, X, Y)
                W2 += 0.5 * omega * np.real(1j * (wf3 * dx_psi2n.conj()) @ w)
                W2 += 0.5 * omega * np.real(1j * (psi2n * dx_wf3.conj()) @ w)
        return W2/Winc

    def _add_wf2(self, n, x, y):
        r"""Compute the additional waveform at an EP2 and its x derivatives.

        The amplitudes \bar{B1} is included.

        Parameters
        ----------
        n : int
            Index of the EP2 modes.
        x : 2D meshgrid array
            x coordinate should vary on the lines.
        y : 2D meshgrid array
            y coordinate should vary on the columns.

        Returns
        -------
        wf2 : array
            The frist term of (23).
        dx_wf2 : array
            The x derivative of the frist term of (23).
        """
        Nmodes = self.Nmodes
        alphan = self.alpha[n]
        sn = np.sqrt(self.k**2 - alphan**2, dtype=complex)
        wf2 = (sn/alphan**2 * self._Chi(y, n)
               - 1j * x * self._Y(y, n))*exp(1j*sn*x)*self.B[Nmodes]
        dx_wf2 = 1j*sn*wf2 - 1j * self._Y(y, n)*exp(1j*sn*x)*self.B[Nmodes]
        return wf2, dx_wf2

    def _add_wf3(self, n, x, y):
        r"""Generate the additional waveform at an EP3 and its x derivatives.

        The amplitudes \bar{\bar{B1}} is included.

        Parameters
        ----------
        n : int
            Index of the EP3 modes.
        x : 2D meshgrid array
            x coordinate should vary on the lines.
        y : 2D meshgrid array
            y coordinate should vary on the columns.

        Returns
        -------
        wf3 : array
            The frist term of (23).
        dx_wf3 : array
            The x derivative of the frist term of (23).
        """
        Nmodes = self.Nmodes
        alphan = self.alpha[n]
        sn = np.sqrt(self.k**2 - alphan**2, dtype=complex)
        Chi = self._Chi(y, n)
        xi = self._xi(y, n)
        Y1 = self._Y(y, n)
        D = alphan**2 + self.mu**2
        wf3 = (- sn**2/alphan**2 * xi
               + self._Ups(n) * Chi
               - 2j*x*(sn/alphan**2 * Chi - (sn*self.mu**2)/(alphan**2*D)*Y1)
               - x**2 * Y1) * exp(1j*sn*x) * self.B[Nmodes+1]
        dx_wf3 = 1j*sn*wf3 + (
                  - 2j*(sn/alphan**2 * Chi - (sn*self.mu**2)/(alphan**2*D)*Y1)
                  - 2*x*Y1) * exp(1j*sn*x) * self.B[Nmodes+1]
        return wf3, dx_wf3

    def _psi2n(self, n, x, y):
        r"""Generate the nth mode and its x derivatives.

        The amplitude Bn is included.

        Parameters
        ----------
        n : int
            Index of the mode.
        x : 2D meshgrid array
            x coordinate should vary on the lines.
        y : 2D meshgrid array
            y coordinate should vary on the columns.

        Returns
        -------
        psi2n : array
            Term from (13).
        dx_psi2n : array
            The x derivative of the term from (13).
        """
        alphan = self.alpha[n]
        sn = np.sqrt(self.k**2 - alphan**2, dtype=complex)
        psi2n = self._Y(y, n) * exp(1j*sn*x) * self.B[n]
        dx_psi2n = 1j*sn*psi2n
        return psi2n, dx_psi2n

    def L(self, m, n, method='ana'):
        r"""Compute \( L_{m,n} = \int_0^1 Y_n(y) \cos(m \pi y) dy \).

        Parameters
        ----------
        m : int
            First mode index.
        n : int
            The second mode index.
        method : string
            Choose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> guide = Guide()
        >>> abs(guide.L(0, 1, 'ana') - guide.L(0, 1, 'num')) < 1e-12
        True

        """
        alphan = self.alpha[n]
        if method == 'ana':
            I = self.p._I1(alphan, m*np.pi) - (self.mu/alphan) * self.p._I2(alphan, m*np.pi)
        elif method == 'num':
            y = self.p.y
            w = self.p.w
            I = (self._Y(y, n) * cos(m*np.pi*y)) @ w
        elif method == 'num2':
            I = (self.p._I1(alphan, m*np.pi, method='num')
                 - (self.mu/alphan) * self.p._I2(alphan, m*np.pi, method='num'))
        else:
            raise ValueError('`method` argument is not recognized.')
        return I

    def M(self, m, n, method='ana'):
        r"""Compute \( M_{m,n} = \int_0^1 \chi_n(y) \cos(m \pi y) dy \).

        Parameters
        ----------
        m : int
            Index of the rigid duct modes.
        n : int
            Index of the double roots.
        method : string
            Choose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> guide = Guide(mu=0., nu=1.650611293539765+2.059981457179885j)
        >>> n = np.where(np.round(guide.multiplicities) == 2)[0].item()
        >>> abs(guide.M(1, n, 'ana') - guide.M(1, n, 'num')) < 1e-12
        True
        >>> abs(guide.M(2, n, 'ana') - guide.M(2, n, 'num')) < 1e-12
        True

        """
        # alphan should be an EP2 (npt tested here)
        alphan = self.alpha[n]
        if method == 'ana':
            mu = self.mu
            I = (- alphan*self.p._I5(alphan, m*np.pi) - mu * self.p._I4(alphan, m*np.pi)
                 + alphan * mu/(alphan**2 + mu**2) * self.p._I2(alphan, m*np.pi)
                 + mu**2/(alphan**2 + mu**2) * self.p._I1(alphan, m*np.pi))
        elif method == 'num':
            y = self.p.y
            w = self.p.w
            I = (self._Chi(y, n) * cos(m*np.pi*y)) @ w
        else:
            raise ValueError('`method` argument is not recognized.')
        return I

    def N(self, m, n, method='ana'):
        r"""Compute \( N_{m,n} = \int_0^1 \xi_n(y) \cos(m \pi y) dy \).

        Parameters
        ----------
        m : int
            Index of the rigid duct modes.
        n : int
            Index of the triple roots.
        method : string
            Choose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> guide = Guide(mu=3.08753629148967+3.62341792246081j, nu=3.17816250726595+4.67518038763374j)
        >>> n = np.where(np.round(guide.multiplicities) == 3)[0].item()
        >>> abs(guide.N(1, n, 'ana') - guide.N(1, n, 'num')) < 1e-12
        True
        >>> abs(guide.N(3, n, 'ana') - guide.N(3, n, 'num')) < 1e-12
        True

        """
        # alphan should be an EP2 (npt tested here)
        alphan = self.alpha[n]
        if method == 'ana':
            mu = self.mu
            D = (alphan**2 + mu**2)
            p = self.p
            # Wrong sign in Eq. (19) Evaluation of LMN_06June2021.pdf
            I = (p._I7(alphan, m*np.pi) - (mu/alphan)*p._I8(alphan, m*np.pi)
                 - 2*mu/D * p._I4(alphan, m*np.pi)
                 + 2*mu**2/(alphan*D) * p._I5(alphan, m*np.pi)
                 + (mu**2/D**2) * self.L(m, n)
                 + (2*mu*alphan/D**2) * p._I2(alphan, m*np.pi)
                 + (2*mu**2/D**2) * p._I1(alphan, m*np.pi))
        elif method == 'num':
            y = self.p.y
            w = self.p.w
            I = (self._xi(y, n) * cos(m * np.pi * y)) @ w
        else:
            raise ValueError('`method` argument is not recognized.')
        return I


class Product():
    """Contains the computation of products of trigonometric functions.

    All methods take an optional `method` argument to specify if the intergral
    is computed by `ana` or `num` (Gauss-Legendre) method.

    Remarks
    -------
    In the present implementation, special cases with equal or with vanishing
    wavenumber are not supported for the analytic version.
    """

    def __init__(self, mu=1, nu=1):
        _NgaussPoint = 50
        self.a = 0
        self.b = 1.
        self.mu = mu
        self.nu = nu
        self.y, self.w = self._gauleg(_NgaussPoint, a=0, b=1)

    @staticmethod
    def _gauleg(N, a=0, b=1):
        """Compute Gauss-Legendre points and weights between [a, b].

        Validation tests
        >>> x, w = Product._gauleg(12, 0, 2*np.pi)
        >>> abs(np.cos(x) @ w) < 1e-12
        True
        >>> abs(np.cos(x)**2 @ w - np.pi) < 1e-11
        True

        """
        # Gauss-Legendre (default interval is [-1, 1])
        x, w = np.polynomial.legendre.leggauss(N)
        # Translate x values from the interval [-1, 1] to [a, b]
        x = 0.5*(x + 1)*(b - a) + a
        w *= 0.5*(b-a)
        return x, w

    def _I1(self, alpha, beta, method='ana'):
        r"""Compute \( \int_0^1 \cos(\alpha y) \cos(\beta y) dy \).

        This integral apears in `Lmn` term.

        Parameters
        ----------
        method : string
            Choose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> p = Product()
        >>> abs(p._I1(0.5+0.2j, 0.98, 'ana') - p._I1(0.5+0.2j, 0.98, 'num')) <1e-12
        True
        """
        if method == 'ana':
            I = (alpha*sin(alpha)*cos(beta) - beta*cos(alpha)*sin(beta))\
                / (alpha**2 - beta**2)
        elif method == 'num':
            I = (np.cos(alpha * self.y) * np.cos(beta * self.y)) @ self.w
        else:
            raise ValueError('`method` argument is not recognized.')
        return I

    def _I2(self, alpha, beta, method='ana'):
        r"""Compute \( \int_0^1 \sin(\alpha y) \cos(\beta y) dy \).

        This integral apears in `Lmn` term.

        Parameters
        ----------
        method : string
            Choose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> p = Product()
        >>> abs(p._I2(0.5+0.2j, 0.98, 'ana') - p._I2(0.5+0.2j, 0.98, 'num')) <1e-12
        True
        """
        if method == 'ana':
            I = -(alpha*cos(alpha)*cos(beta) + beta*sin(alpha)*sin(beta))\
                / (alpha**2 - beta**2) + alpha/(alpha**2 - beta**2)
        elif method == 'num':
            I = (np.sin(alpha * self.y) * np.cos(beta * self.y)) @ self.w
        else:
            raise ValueError('`method` argument is not recognized.')
        return I

    def _I3(self, alpha, beta, method='ana'):
        r"""Compute \( \int_0^1 \sin(\alpha y) \sin(\beta y) dy \).

        This integral apears in `Mmn` term.

        Parameters
        ----------
        method : string
            Choose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> p = Product()
        >>> abs(p._I3(0.5+0.2j, 0.98, 'ana') - p._I3(0.5+0.2j, 0.98, 'num')) <1e-12
        True
        """
        if method == 'ana':
            I = (-alpha*cos(alpha)*sin(beta) + beta*sin(alpha)*cos(beta))\
                / (alpha**2 - beta**2)
        elif method == 'num':
            I = (np.sin(alpha * self.y) * np.sin(beta * self.y)) @ self.w
        else:
            raise ValueError('`method` argument is not recognized.')
        return I

    def _I4(self, alpha, beta, method='ana'):
        r"""Compute \( \int_0^1 y \cos(\alpha y) \cos(\beta y) dy \).

        This integral apears in `Mmn` term.

        Parameters
        ----------
        method : string
            Choose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> p = Product()
        >>> abs(p._I4(0.5+0.2j, 0.98, 'ana') - p._I4(0.5+0.2j, 0.98, 'num')) <1e-12
        True
        """
        if method == 'ana':
            I = self._I1(alpha, beta) \
                - (alpha/(alpha**2-beta**2)) * self._I2(alpha, beta) \
                + (beta/(alpha**2-beta**2)) * self._I2(beta, alpha)
        elif method == 'num':
            I = (self.y * np.cos(alpha * self.y) * np.cos(beta * self.y)) @ self.w
        else:
            raise ValueError('`method` argument is not recognized.')
        return I

    def _I5(self, alpha, beta, method='ana'):
        r"""Compute \( \int_0^1 y \sin(\alpha y) \cos(\beta y) dy \).

        This integral apears in `Mmn` term.

        Parameters
        ----------
        method : string
            Choose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> p = Product()
        >>> abs(p._I5(0.5+0.2j, 0.98, 'ana') - p._I5(0.5+0.2j, 0.98, 'num')) <1e-12
        True
        """
        if method == 'ana':
            I = self._I2(alpha, beta) - alpha/(alpha**2-beta**2) \
                + (alpha/(alpha**2-beta**2)) * self._I1(alpha, beta) \
                + (beta/(alpha**2-beta**2)) * self._I3(beta, alpha)
        elif method == 'num':
            I = (self.y * np.sin(alpha * self.y) * np.cos(beta * self.y)) @ self.w
        else:
            raise ValueError('`method` argument is not recognized.')
        return I

    def _I6(self, alpha, beta, method='ana'):
        r"""Compute \( \int_0^1 y \sin(\alpha y) \sin(\beta y) dy \).

        This integral apears in `Mmn` term.

        Parameters
        ----------
        method : string
            Choose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> p = Product()
        >>> abs(p._I6(0.5+0.2j, 0.98, 'ana') - p._I6(0.5+0.2j, 0.98, 'num')) <1e-12
        True
        """
        if method == 'ana':
            I = self._I3(alpha, beta) \
                + (alpha/(alpha**2-beta**2)) * self._I2(beta, alpha) \
                - (beta/(alpha**2-beta**2)) * self._I2(alpha, beta)
        elif method == 'num':
            I = (self.y * np.sin(alpha * self.y) * np.sin(beta * self.y)) @ self.w
        else:
            raise ValueError('`method` argument is not recognized')
        return I

    def _I7(self, alpha, beta, method='ana'):
        r"""Compute \( \int_0^1 y**2 \cos(\alpha y) \cos(\beta y) dy \).

        This integral apears in `Mmn` term.

        Parameters
        ----------
        method : string
            Choose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> p = Product()
        >>> abs(p._I7(0.5+0.2j, 0.98, 'ana') - p._I7(0.5+0.2j, 0.98, 'num')) <1e-12
        True
        """
        if method == 'ana':
            I = self._I1(alpha, beta) \
                - (2*alpha/(alpha**2-beta**2)) * self._I5(alpha, beta) \
                + (2*beta/(alpha**2-beta**2)) * self._I5(beta, alpha)
        elif method == 'num':
            I = (self.y**2 * np.cos(alpha * self.y) * np.cos(beta * self.y)) @ self.w
        else:
            raise ValueError('`method argument is not recognized.')
        return I

    def _I8(self, alpha, beta, method='ana'):
        r"""Compute \( \int_0^1 y**2 \sin(\alpha y) \cos(\beta y) dy \).

        This integral apears in `Mmn` term.

        Parameters
        ----------
        method : string
            Choose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> p = Product()
        >>> abs(p._I8(0.5+0.2j, 0.98, 'ana') - p._I8(0.5+0.2j, 0.98, 'num')) <1e-12
        True
        """
        if method == 'ana':
            I = self._I2(alpha, beta) - alpha/(alpha**2-beta**2)\
                + (2*alpha/(alpha**2-beta**2)) * self._I4(alpha, beta) \
                + (2*beta/(alpha**2-beta**2)) * self._I6(alpha, beta)
        elif method == 'num':
            I = (self.y**2 * np.sin(alpha * self.y) * np.cos(beta * self.y)) @ self.w
        else:
            raise ValueError('`method` argument is not recognized.')
        return I


# %% Filters classes
class Filters(ABC):
    """Define the abstract class for enhancing the convergence of the Fourier series with filters.

    Based on
    - Gottlieb, David, and Chi-Wang Shu. "On the Gibbs phenomenon and
      its resolution." SIAM review 39.4 (1997): 644-668.
    - Nawaz, Rab, and Jane B. Lawrie. "Scattering of a fluid-structure coupled
      wave at a flanged junction between two flexible waveguides."
      The Journal of the Acoustical Society of America 134.3 (2013): 1939-1949.
    """

    def __init__(self, N):
        """Initialize the filter with `N`, the number of terms in the Fourier serie."""
        self.N = N - 1

    def name(self):
        """Return filter name."""
        return self.__class__.__name__.replace('Filter', '')

    @abstractmethod
    def filter(self):
        """Return the sigma factors."""
        pass


class NoneFilter(Filters):
    """No filter case, sigmas are just ones."""

    def filter(self):
        """Return unit sigma factors."""
        return np.ones(self.N+1)


class LanczosFilter(Filters):
    """Implement the Lanczos sigma filter."""

    def filter(self):
        """Return the sigma factors."""
        a = np.arange(0, self.N+1) / self.N
        # pi is already in sinc
        sigma = np.sinc(a)
        return sigma


class RaisedCosineFilter(Filters):
    """Implement the Raised Cosine sigma filter (order 2)."""

    def filter(self):
        """Return the sigma factors."""
        a = np.arange(0, self.N+1) * np.pi / self.N
        sigma = 0.5 * (1 + np.cos(a))
        return sigma


class SharpenedRaisedCosineFilter(Filters):
    """Implement the Sharpened Raised Cosine sigma filter (order 8)."""

    def __init__(self, N):
        self._filter = RaisedCosineFilter(N).filter
        self.N = N

    def filter(self):
        """Return the sigma factors."""
        sigma3 = self._filter()
        sigma = sigma3**4 * (35. - 84.*sigma3 + 70.*sigma3**2 - 20.*sigma3**3)
        return sigma


CreateFilter = {'SharpenedRaisedCosine': SharpenedRaisedCosineFilter,
                'RaisedCosine': RaisedCosineFilter,
                'Lanczos': LanczosFilter,
                'None': NoneFilter}
"""
Factory method to create basic filter. The keys are the filter name without
the 'Filter' suffix.
"""
