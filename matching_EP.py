#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Mode matching at an EP.

based on J. Lawrie work.
Created on Wed Jul  7 10:04:58 2021

implement matching
  - [] No EP
  - [] EP2
  - [] EP3


Basic post-process
  - [] plot against the frequency

@author: bn
"""
import numpy as np
import polze
from matplotlib import pyplot as plt
cos, sin = np.cos, np.sin


class Guide():
    """ Define a semi-infinite waveguide with a rigid duct on the right side
    (x<0) and a lined duct (x>0) on the left side.
    `mu` and `nu` denotes the two wall admittances respectivelly on the bottom
    (y=0) and on the top (y=1).
    """

    def __init__(self, mu=1, nu=1j, k=1, Nmodes=2, R=5*3.14):

        self.a = 0.
        self.b = 1.
        self.mu = mu
        self.nu = nu
        self.k = k
        self.Nmodes = Nmodes
        alpha = self.solve_disp(R)
        self.p = Product(mu, nu)

    def _K(self, alpha):
        """ Dispersion equation express in alpha.
        """
        mu, nu = self.mu, self.nu
        return (mu+nu)*cos(alpha) + sin(alpha)*(alpha - mu*nu/alpha)

    def _Kp(self, alpha):
        """ Derivative with respect to s of the dispersion equation.
        Evaluate at alpha.
        """
        mu, nu = self.mu, self.nu
        s = np.sqrt(self.k**2 - alpha**2, dtype=complex)
        p = mu*nu
        gp = cos(alpha)/alpha - sin(alpha)/alpha**2
        Kp = (-s/alpha)*((1-nu-mu)*sin(alpha) + alpha*cos(alpha) - gp*p)
        return Kp

    def _Y(self, y, n):
        """ Compute transverse field.
        """
        alphan = self.alpha[n]
        Yn = cos(alphan * y) - self.mu*sin(alphan * y)/alphan
        return Yn

    def P(self, n, method='ana'):
        """ Compute modal norm.

        Parameters
        ----------
        method : string
            chose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> guide = Guide(1., 2j)
        >>> alpha = guide.solve_disp(R=6)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        >>> abs(guide.P(0, method='ana') - guide.P(0, method='num')) < 1e-10
        True
        """
        if method == 'ana':
            alphan = self.alpha[n]
            sn = np.sqrt(self.k**2 - alphan**2, dtype=complex)
            P = - self._Y(1, n) / (2*sn)  * self._Kp(alphan)
        elif method == 'num':
            y = self.p.y
            Yn = self._Y(y, n)
            P = (Yn * Yn) @ self.p.w
        else:
            raise ValueError('`method argument is not recognized')
        return P

    def solve_disp(self, R=10, Npz=30, Ni=15000, display=False):
        """ Use contour integral to solve the dispersion equation.

        Examples
        --------
        Check on triple roots case.
        >>> guide = Guide(3.1781625072659+4.67518038763374j, 3.08753629148967+3.62341792246081j)
        >>> np.abs(guide.alpha - (4.19693888263064 - 2.60864153789443j) ).min() < 1e-10
        True
        """
        pz = polze.PZ(self._K, Rmax=R, Npz=Npz, Ni=Ni, split=True,
                      options={'_vectorized': True, '_Npz_limit': 10})
        # solve with moment method
        pz.solve()
        if display:
            pz.display()
        _, alphan = pz.dispatch()
        # all modes come by pair...
        alphan = alphan[alphan.real>0]
        # return sorted in comlex lexicographic order
        alpha = np.sort_complex(alphan)
        self.alpha = alpha
        # TODO store also multiplicity...
        return alpha

    @staticmethod
    def eps(n):
        """ Return 2 if n==0 and 1 otherwise.
        """
        if n==0:
            return 2
        else:
            return 1


    def matching_std(self):
        """ Compute the reflected and the transmitted waves when there is no
        EP.

        The incident field is a plane wave.
        A and B denotes the refected and the transmitted waves respectivelly.
        """
        Nmodes = self.Nmodes
        rhs = np.zeros((2*Nmodes,), dtype=complex)
        # Eq. (16)
        I = np.eye(Nmodes)
        MB = np.zeros((Nmodes, Nmodes), dtype=complex)
        for (m, n), bmn in np.ndenumerate(MB):
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
        return A, B

    def plot_field(self):
        """ Plot potential field to check the continuity.
        """
        N = 50
        F = 1.
        x1, y1 = np.meshgrid(np.linspace(-1, 0, N), np.linspace(0, 1, N))
        x2, y2 = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
        
        Psi1 = F*np.exp(1j*self.k*x1)
        Psi2 = np.zeros_like(x2, dtype=complex)
        for n in range(0, self.Nmodes):
            sn = np.sqrt(self.k**2 - self.alpha[n]**2, dtype=complex)
            etan = np.sqrt(self.k**2 - (n*np.pi)**2, dtype=complex)
            Psi1 += self.A[n] * np.exp(-1j*etan*x1) * cos(n * np.pi * y1)
            Psi2 += self.B[n] * np.exp(1j*sn*x2) * self._Y(y2, n)
            
        fig = plt.figure()
        ax = plt.gca()
        ax.pcolormesh(x1, y1, np.abs(Psi1), shading='auto', vmin=-2, vmax=2)
        ax.pcolormesh(x2, y2, np.abs(Psi2), shading='auto', vmin=-2, vmax=2)
        
        fig1d = plt.figure('in the midle of the duct')
        plt.plot(x1[N//2, :], Psi1[N//2, :].real)
        plt.plot(x2[N//2, :], Psi2[N//2, :].real)
        # plt.colorbar(contour_F, ax=ax)

    def L(self, m, n, method='ana'):
       r""" Compute \( L_{m,n} = \int_0^1 Y_n(y) \cos(m \pi y) dy \).

       Parameters
       ----------
       method : string
           chose a method to compute the integral. Should be in {'ana', 'num'}.

       Examples
       --------
       >>> guide = Guide()
       >>> abs(guide.L(0, 1, 'ana') - guide.L(0, 1, 'num')) <1e-12
       True

       """
       alphan = self.alpha[n]
       if method == 'ana':
           I = self.p._I1(alphan, m*np.pi) - (self.mu/alphan) * self.p._I2(alphan, m*np.pi)
       elif method == 'num':
           I = (self.p._I1(alphan, m*np.pi, method='num')
                - (self.mu/alphan) * self.p._I2(alphan, m*np.pi, method='num'))
       else:
           raise ValueError('`method` argument is not recognized.')
       return I

class Product():
    """ Contains the computation of products of trigonometric functions.

    All methods take an optional `method` argument o specify if the intergral
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
        """ Compute Gauss-Legendre points and weights between [a, b].

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
        r""" Compute \( \int_0^1 \cos(\alpha y) \cos(\beta y) dy \).

        This integral apears in `Lmn` term.

        Parameters
        ----------
        method : string
            chose a method to compute the integral. Should be in {'ana', 'num'}.

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
        r""" Compute \( \int_0^1 \sin(\alpha y) \cos(\beta y) dy \).

        This integral apears in `Lmn` term.

        Parameters
        ----------
        method : string
            chose a method to compute the integral. Should be in {'ana', 'num'}.

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
        r""" Compute \( \int_0^1 \sin(\alpha y) \sin(\beta y) dy \).

        This integral apears in `Mmn` term.

        Parameters
        ----------
        method : string
            chose a method to compute the integral. Should be in {'ana', 'num'}.

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
        r""" Compute \( \int_0^1 y \cos(\alpha y) \cos(\beta y) dy \).

        This integral apears in `Mmn` term.

        Parameters
        ----------
        method : string
            chose a method to compute the integral. Should be in {'ana', 'num'}.

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
        r""" Compute \( \int_0^1 y \sin(\alpha y) \cos(\beta y) dy \).

        This integral apears in `Mmn` term.

        Parameters
        ----------
        method : string
            chose a method to compute the integral. Should be in {'ana', 'num'}.

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
        r""" Compute \( \int_0^1 y \sin(\alpha y) \sin(\beta y) dy \).

        This integral apears in `Mmn` term.

        Parameters
        ----------
        method : string
            chose a method to compute the integral. Should be in {'ana', 'num'}.

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
        r""" Compute \( \int_0^1 y**2 \cos(\alpha y) \cos(\beta y) dy \).

        This integral apears in `Mmn` term.

        Parameters
        ----------
        method : string
            chose a method to compute the integral. Should be in {'ana', 'num'}.

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
        r""" Compute \( \int_0^1 y**2 \sin(\alpha y) \cos(\beta y) dy \).

        This integral apears in `Mmn` term.

        Parameters
        ----------
        method : string
            chose a method to compute the integral. Should be in {'ana', 'num'}.

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

# %% Main
if __name__ == '__main__':
    import doctest
    # run docttest
    # doctest.testmod(optionflags=doctest.ELLIPSIS, verbose=False)
    # p = Product()
    # abs(p._I4(0.5+0.2j, 0.98, 'ana') - p._I4(0.5+0.2j, 0.98, 'num')) <1e-12
    
    guide = Guide(mu=0, nu=0.1, Nmodes=4, k=10, R=8*3.14)
    # expand the radius to look for alpha
    alpha = guide.solve_disp()
    A, B = guide.matching_std()
    guide.plot_field()