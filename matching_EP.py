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

@author: bn
"""
import numpy as np
cos, sin = np.cos, np.sin

class Product():
    """ Contains the computation of products of trigonometric functions.
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