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

cos, sin = np.cos, np.sin


class Guide():
    """ Define a semi-infinite waveguide with a rigid duct on the right side
    (x<0) and a lined duct (x>0) on the left side.
    `mu` and `nu` denotes the two wall admittances respectivelly on the bottom
    (y=0) and on the top (y=1).
    """

    def __init__(self, mu=1, nu=1, k=1):

        self.a = 0.
        self.b = 1.
        self.mu = mu
        self.nu = nu
        self.k = 1
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
        s = np.sqrt(self.k**2 - alpha**2)
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
        
    def _P(self, n, method='ana'):
        """ Compute modal norm.
        
        Parameters
        ----------
        method : string
            chose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> guide = Guide(1., 2j)
        >>> alpha = guide.solve_disp(R=6)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        <BLANKLINE>
        ...
        >>> abs(guide._P(0, method='ana') - guide._P(0, method='num')) < 1e-10
        True
        """
        if method == 'ana':
            alphan = self.alpha[n]
            sn = np.sqrt(self.k**2 - alphan**2)
            P = - self._Y(1, n) / (2*sn)  * self._Kp(alphan)
        elif method == 'num':
            y = self.p.y
            Yn = self._Y(y, n)
            P = (Yn * Yn) @ self.p.w
        else:
            raise ValueError('`method argument is not reconized')
        return P

    def solve_disp(self, R=10, Npz=20, Ni=5000):
        """ Use contour integral to solve the dispersion equation.

        Examples
        --------
        Check on triple roots case.
        >>> guide = Guide(3.17816250726595 +  4.67518038763374j, 3.08753629148967 +  3.62341792246081j)
        >>> alpha = guide.solve_disp(R=6)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        <BLANKLINE>
        ...
        >>> abs(alpha[1] - (4.19693888263064 - 2.60864153789443j) ) < 1e-10
        True
        """
        pz = polze.PZ(self._K, Rmax=R, Npz=Npz, Ni=Ni, split=True,
                      options={'_vectorized': True, '_Npz_limit': 10})
        # solve with moment method
        pz.solve()
        pz.display()
        _, alphan = pz.dispatch()
        # return sorted in comlex lexicographic order
        alpha = np.sort_complex(alphan)
        self.alpha = alpha
        # TODO store also multiplicity...
        return alpha



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
        # need to provide it :)
        self.alpha = np.arange(0, 4)
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
            raise ValueError('`method argument is not reconized')
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
            raise ValueError('`method argument is not reconized')
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
            raise ValueError('`method argument is not reconized')
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
            raise ValueError('`method argument is not reconized')
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
            raise ValueError('`method argument is not reconized')
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
            raise ValueError('`method argument is not reconized')
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
            raise ValueError('`method argument is not reconized')
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
            raise ValueError('`method argument is not reconized')
        return I

    def L(self, m, n, method='ana'):
        r""" Compute \( L_{m,n} = \int_0^1 Y_n(y) \cos(m \pi y) dy \).

        Parameters
        ----------
        method : string
            chose a method to compute the integral. Should be in {'ana', 'num'}.

        Examples
        --------
        >>> p = Product()
        >>> abs(p._I2(0.5+0.2j, 0.98, 'ana') - p._I2(0.5+0.2j, 0.98, 'num')) <1e-12
        True

        Need to add test with GL
        """
        if method == 'ana':
            alphan = self.alpha[n]
            I = self._I1(alphan, m*np.py) - (self.mu/alphan) * self._I2(alphan, m*np.py)
        elif method == 'num':
            I = (self._I1(alphan, m*np.py, method='num')
                 - (self.mu/alphan) * self._I2(alphan, m*np.py, method='num'))
        else:
            raise ValueError('`method argument is not reconized')
        return I


if __name__ == '__main__':
    import doctest

    # run docttest
    doctest.testmod(optionflags=doctest.ELLIPSIS, verbose=False)
    # p = Product()
    # abs(p._I4(0.5+0.2j, 0.98, 'ana') - p._I4(0.5+0.2j, 0.98, 'num')) <1e-12
