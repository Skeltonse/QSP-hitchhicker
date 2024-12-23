#!/usr/bin/env python
# coding: utf-8

# Functions to check Laurent polynomials. All coefficient lists are numpy arrays ordered $\left(c_{-n}, c_{-n+1},...c_0, ...c_n\right)$. Default tolerance is $10^{-16}$ unless otherwise specified
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def REAL_CHECK(coeff, n, theta=np.linspace(-np.pi,np.pi, 100),  tol=10**(-16), fcnname='function', giveprecision=False):
    """
    CHECKS IF A LAURENT POLY IS REAL-ON-CIRLCE.
    Computes values on the unit circle, for Laurent polynomial with coefficients a of degree n.
    returns an error if any are larger than the set tolerance

    inputs:
    coeff: length 2n+1 np array, coefficient list of a Laurent polynomial
    n: float, degree of the Laurent polynomial
    theta: np array of points to check functional values
    tol: float, tolerance of solution
    fcnname: string naming the function being checked
    giveprecision: True/False option to return the max error in the Laurent polynomial

    return:
    coeffQ: np array of function values, Laurent polynomial evaluated at each point in theta
    
    """
    coeffQ=LAUR_POLY_BUILD(coeff, n,  np.exp(1j*theta))
    
    if max(abs(np.imag(coeffQ)))>tol:
        print('warning, '+ fcnname +' has nontrivial imaginary component')
        print("largest imaginary component is", max(abs(np.imag(coeffQ))))
        if giveprecision==True:
            return coeffQ, max(abs(np.imag(coeffQ)))
    else:
        if giveprecision==True:
            return coeffQ, tol
    return coeffQ

def CHEBY_POLY_BUILD(coeff, n, th, term='c'):
    """
    computes the float-point value of a Fourier expansion from its Chebyshev coefficient list.

    inputs:
    coef: length n+1 coefficient array
    n: float, degree of polynomial
    theta: float or np array of points in \theta\in (-\pi, \pi) to check functional values
    term: string, determines whether the cosinal or sinusoidal term is being computed
    --'s': computes the sin expansion
    --'c' computes the cosine expansion
    """
    polyval=0
    if term=='c':
        for l in range(0, n+1):
            polyval=polyval+coeff[l]*np.cos(l*th)
    elif term=='s':
        for l in range(0, n+1):
            polyval=polyval+coeff[l]*np.sin(l*th)
    return polyval

def GET_LAURENT(clist, slist, n):
    """
    Converts the coefficient list of polynomials in x\in[-1, 1] to the coefficent list of the corresponding Laurent polynomial

    inputs:
    clist: length n+1 array, the coefficeint list of even-power terms in the polynomial
    slist: length n+1 array, the coefficeint list of odd-power terms in the polynomial
    n: degree of the polynomial
    """
    czlist=np.append(np.append(np.flip(clist[1:])/2,  [clist[0]]),clist[1:]/2)
    szlist=np.append(np.append(-np.flip(slist[1:])/2j,  [slist[0]]),slist[1:]/2j)
    return czlist, szlist

def LAUR_POLY_MULT(coeffa, coeffb):
    """
    produces the coefficient list of the prodcut of polynomials a, b from their coefficient lists.
    inputs:
    coeffa, coeffb: length 2n+1 np arrays, the coefficient lists of two polynomials

    return: coefficient list of product polynomial
    """
    return sig.convolve(coeffa, coeffb, method='fft')

def LAUR_POLY_BUILD(coeff, n, z):
    """
    computes the float-point value of a Laurent polynomial from its coefficient list (does not assume symmetric coefficents).
    
    inputs:
    coef: length n+1 coefficient array
    n: float, degree of polynomial
    z: float or np array of points to check functional values

    return:
    float or np array of functional values
    """
    polyval=0
    for l in range(-n,n+1):
        polyval=polyval+coeff[l+n]*z**l
    return polyval

def POLY_BUILD(coeff, n, z):
    """computes the float-point value of a polynomial from its coefficient list.
   inputs:
    coef: length n+1 coefficient array
    n: float, degree of polynomial
    z: float or np array of points to check functional values

    return:
    float or np array of functional values
    """
    polyval=0
    for l in range(0,n+1):
        polyval=polyval+coeff[l]*z**l
        
    return polyval
